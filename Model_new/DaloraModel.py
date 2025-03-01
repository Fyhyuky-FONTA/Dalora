import torch
import torch.nn as nn

from typing import Any, Dict, List
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaConfig

from .DaloraConfig import DaloraConfig

# 获取当前模块的完整名称
def get_module(model, name):
    '''
    :param name: str module name
    :return: nn.Module, current module
    '''
    for na, module in model.named_modules():
        if na == name:
            return module

'''
这里需要建立模型的映射来加载原始模型，首先我们利用Llama的config文件初始化一个空的Llama模型，然后为了与保存模型的状态字典相同，
我们将空白模型的所有层按照状态字典的键值进行初始化，所有属性与原始模型的属性相同，并且原始
'''

# 模型映射
class DaloraLinear(nn.Module):
    r"""
    active_adapters (Union[List[`str`], `str`], *optional*): The name of the active adapter.
    """
    # indicates whether all adapters should be disabled
    _disable_adapters: bool = False

    # List all merged adapters
    merged_adapters: list[str] = []

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("Dalora_A", "Dalora_B")

    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "Dalora_alpha", "scaling", "Dalora_dropout")

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.base_layer.requires_grad = False

        # 从config文件中读取，最后一步进行重构，因为需要得到所有Adapter的名字
        self.active_adapters: list[str] = []
        self.r = {}
        self.scaling = {}
        self.Dalora_dropout = nn.ModuleDict({})

        # 空白，等待重构
        self.Dalora_A = nn.ModuleDict({})
        self.Dalora_B = nn.ModuleDict({})

    # 重构Dalora层
    def Rebuild(self, adapter_name, layer_name, rank):
        '''
        :param adapter_name: adapter name
        :param layer_name: the name of the current layer under rebuilding
        :param rank: DaLoRA rank
        '''
        if layer_name == 'Dalora_A':
            self.Dalora_A[adapter_name] = nn.Linear(self.in_features, rank, bias=False)
            self.Dalora_A[adapter_name].requires_grad = True
        elif layer_name == 'Dalora_B':
            self.Dalora_B[adapter_name] = nn.Linear(rank, self.out_features, bias=False)
            self.Dalora_B[adapter_name].requires_grad = True

    # 按照Adapter Linear层进行编写
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.Dalora_A.keys():
                continue
            Dalora_A = self.Dalora_A[active_adapter]
            Dalora_B = self.Dalora_B[active_adapter]
            dropout = self.Dalora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            x = x.to(Dalora_A.weight.dtype)
            result = result + Dalora_B(Dalora_A(dropout(x))) * scaling
        result = result.to(torch_result_dtype)

        return result

class DaloraLlamaForCausalLM(LlamaForCausalLM):
    # 这里指定当前模型的config对象，额外的对象属性保存在特定路径下的json对象中，
    # 在创建config对象时将会根据json文件创建出这些添加的属性
    config_class = DaloraConfig

    def __init__(self, config: DaloraConfig):
        super().__init__(config)  # 使用config文件首先初始化一个Llama模型

        '''
        module_dict: dict, saved model's state dict
        extra_dict: other useful parameter with format of:
            {
                adapter_name1: {
                    "dropout": dropout_prob,
                    "rank": rank,
                    "scaling": scaling
                }
                adapter_name2: {
                    "dropout": dropout_prob,
                    "rank": rank,
                    "scaling": scaling
                }
            }
        '''

        # 这里从config对象中取出两个字典
        module_dict = config.module_dict
        extra_dict = config.extra_dict

        # 层序遍历所有模块
        full_name_dict = {module: name for name, module in self.named_modules()}
        linear_info = {}
        modules = [self]
        while len(modules) > 0:
            submodule = modules.pop()
            for name, raw_linear in submodule.named_children():
                if isinstance(raw_linear, nn.Linear):
                    full_name = full_name_dict[raw_linear]
                    linear_info[raw_linear] = {
                        "father": submodule,
                        "name": name,
                        "full_name": full_name,
                    }
                else:
                    modules.append(raw_linear)

        # 遍历线性层模块名称，进行层重构
        for linear in module_dict:
            if 'Dalora' in linear:
                # 获取'Dalora'之前的所有字符串并找到对应的模块
                layer = linear[:linear.find(".Dalora")]
                module = get_module(self, layer)
                info = linear_info[module]

                # 如果是线性层则进行创建
                if isinstance(module, nn.Linear):
                    Layer = DaloraLinear(
                        module.in_features,
                        module.out_features,
                    )
                    setattr(info["father"], info["name"], Layer)

                    del linear_info[module]  # 删除原始键值
                    module = get_module(self, layer)  # 更新

                    # 更新字典
                    linear_info[module] = {
                        "father": info["father"],
                        "name": info["name"],
                        "full_name": layer,
                    }

                # 找到层名称标识
                linear = linear.split('.')
                index = linear.index(info['name']) + 1
                adapter_name = linear[index + 1]  # adapter name, like 'default'
                layer_name = linear[index]  # Dalora_A or Dalora_B

                module.Rebuild(
                    adapter_name=adapter_name,
                    layer_name=layer_name,
                    rank=extra_dict[adapter_name]['rank']
                )

            # 将一般线性层的梯度关闭
            else:
                module = get_module(self, linear[:linear.find(".weight")])
                if isinstance(module, nn.Linear):
                    module.requires_grad = False

        # 遍历所有模块名称，使用额外参数进行参数补全
        for name, module in self.model.named_modules():
            if isinstance(module, DaloraLinear):
                module.active_adapters = extra_dict.keys()
                for adapter in extra_dict.keys():
                    module.r[adapter] = extra_dict[adapter]['rank']
                    module.scaling[adapter] = extra_dict[adapter]['scaling']
                    module.Dalora_dropout[adapter] = nn.Dropout(p=extra_dict[adapter]['dropout'])