import logging
import os

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, List

import torch
import torch.nn as nn
import numpy as np
import re
from itertools import chain
import math
import time
from tqdm import tqdm

from Adapter.utils import (
    _ExcludedModule,
    _get_submodules,
    check_target_module_exists,
    _prepare_4d_causal_attention_mask_with_cache_position,
    apply_rotary_pos_emb,
    repeat_kv,
    get_name
)
from Adapter.config import DaLoraConfig
from Adapter.layer import DaLoraLayer, Linear

logger = logging.getLogger(__name__)

class DaLoraBuilder(nn.Module):
    r"""
    Attributes:
        -model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        -Dalora_config (`Union[`PeftConfig`, dict[str, PeftConfig]]`):The adapter configuration object, it should be
        a dictionary of `str` to `PeftConfig` objects. One can also pass a PeftConfig object and a new adapter will
        be created with the default name `adapter` or create a new dictionary with a key `adapter_name` and a value
        of that peft config.
        -config (`dict[str, Any]`): The model configuration object, it should be a dictionary of `str` to `Any` objects.
        -targeted_module_names (`list[str]`):The list of module names that were actually adapted. Can be useful to
        inspect if you want to quickly double-check that the `config.target_modules` were specified correctly.
    """

    prefix: str = "Dalora_"

    def __init__(
        self,
        tokenizer,
        model,
        sample_X: List[str],
        Dalora_config: Union[DaLoraConfig, dict[str, DaLoraConfig]],
        adapter_name: str,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer  # tokenizer of model to add adapter to
        self.model = model  # transformers model to add adapter to
        self.data = sample_X  # sampled data

        self.targeted_module_names: list[str] = []  # module name (linear layer name) to add adapter

        # init
        if not hasattr(self, "Dalora_config"):
            self.Dalora_config = {}
        self.Dalora_config[adapter_name] = Dalora_config  # config of current adapter (a set of DaLoRA layer)

        self.active_adapter: str | list[str] = adapter_name  # adapter name
        self.model.Dalora_config = self.Dalora_config  # Copy the Dalora_config in the injected model.

        # inject adapter into all module which need to add adapter to
        self.inject_adapter(self.model, adapter_name)

    @property
    def active_adapters(self) -> list[str]:
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    # 将向各层注入adapter，注意需要将原始数据逐渐通过各个层
    def inject_adapter(
        self,
        model: nn.Module,
        adapter_name: str,
        autocast_adapter_dtype: bool = True,
    ) -> None:
        r"""
            Creates adapter layers and replaces the target modules with the adapter layers.
            This method is called under the hood by `peft.mapping.get_peft_model` if a non-prompt
            tuning adapter class is passed.
            The corresponding PEFT config is directly retrieved from the `Dalora_config` attribute of
            the BaseTuner class.
            Args:
                model (`nn.Module`): The model to be tuned.
                adapter_name (`str`): The adapter name.
        """
        Dalora_config = self.Dalora_config[adapter_name]  # 当前adapter的配置文件
        device = model.device

        '''
            由于我们使用的是left padding，于是mask向量开头会有前导0，这意味着这部分的语义聚合是
            空的，于是前导零对应的输出向量全部都是Nan值，这和right padding不同，right padding末
            尾的padding token对应的张量仍然能够聚合之前的token信息，不过聚合的长度都是相同的，于
            是结尾的padding token输出张量不为Nan值，但是输出结果都相同。
            为了处理left padding导致的Nan值问题，我们可以在每一层构建adapter之前将Nan值替换为0，
            这样能够在使用SVD逼近的过程中使得这些位置被忽略掉。
        '''
        '''<<< 下面代码需要根据模型类型进行改写，代码写法对于不同的模型存在一定的区别，这里针对的是Llama-3-8B模型编写 >>>'''

        # 对输入进行padding，这里使用left padding，并且设置padding token为开始符号
        self.tokenizer.pad_token = self.tokenizer.eos_token  # padding token设置为开始token
        self.tokenizer.padding_side = "left"  # 设置为left padding，这样需要更改mask及其余辅助矩阵的规则
        X = self.tokenizer(self.data, return_tensors="pt", padding=True, truncation=True)  # 编码
        input_ids = X["input_ids"].to(device)  # 需要将类型转化为long

        # 过Embedding层
        EmbedLayer = self.model.model.embed_tokens  # Embedding Layer
        inputs_embeds = EmbedLayer(input_ids)  # [B, L, din]，数据类型为bfoat16
        batch, seq, _ = inputs_embeds.shape

        # 创建mask矩阵（每个注意力头分配一组，数据类型bfoat16
        attention_mask = X["attention_mask"].to(device)

        # print(attention_mask)

        dtype = inputs_embeds.dtype  # 获取Embedding张量的信息
        target_length = attention_mask.shape[-1]
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)  # cache位置参数
        attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask=attention_mask,
           sequence_length=seq,
           target_length=target_length,
           dtype=dtype,
           device=device,
           cache_position=cache_position,
           batch_size=inputs_embeds.shape[0]
        )

        # 使用虚张量获取旋转位置编码张量，对照源码进行编写
        position_ids = cache_position.unsqueeze(0)
        lama_model = self.model.model
        attn = lama_model.layers[0].self_attn  # 获取第一层的attention层即可
        virtual = torch.rand(batch, seq, attn.num_key_value_heads, attn.head_dim).transpose(1, 2).to(dtype)  # 注意需要进行类型转换
        position_embeddings = attn.rotary_emb(virtual, position_ids)

        # 将数据张量经过网络的每一个模块
        with torch.no_grad():
            for layer_idx in tqdm(range(len(lama_model.layers))):
                '''
                    Copy from Attention module forward function down here, we catch each of the sub-module's input 
                    and output to construct our B & A matrix. It is important to note that we should pass the sampling
                    data through the layer which we want to add adapter before adding adapter to it to in case the 
                    layer is changed to lead some error.
                '''
                Decoder = lama_model.layers[layer_idx]

                # 提前获取下一层输出，不返回attention输出和Cache输出模式
                # 将Nan值置为0
                nexts_embeds = Decoder(
                    hidden_states=inputs_embeds,
                   attention_mask=attention_mask,
                   position_ids=position_ids,
                   past_key_value=None,
                   output_attentions=False,
                   use_cache=False,
                   cache_position=cache_position,
                   position_embeddings=position_embeddings
                )[0]
                nexts_embeds = torch.nan_to_num(nexts_embeds, nan=0.0)
                
                ''' Attention Layer '''
                attention = Decoder.self_attn
                XRMS = Decoder.input_layernorm(inputs_embeds)  # RMS归一化层，是attention层的输入

                # prepare data for next layer first
                query_states = attention.q_proj(XRMS)  # [B, seq, attn_head * head_dim]
                key_states = attention.k_proj(XRMS)  # [B, seq, KV_head * head_dim]
                value_states = attention.v_proj(XRMS)  # [B, seq, KV_head * head_dim]

                # Attention
                query_states = query_states.view(batch, seq, attention.num_heads, attention.head_dim).transpose(1, 2)
                key_states = key_states.view(batch, seq, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
                value_states = value_states.view(batch, seq, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)

                # positional embedding
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                # repeating for GQA
                key_states = repeat_kv(key_states, attention.num_key_value_groups)
                value_states = repeat_kv(value_states, attention.num_key_value_groups)

                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attention.head_dim)

                # mask
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

                attn_weights = attn_weights + causal_mask

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights = nn.functional.dropout(attn_weights, p=attention.attention_dropout, training=attention.training)

                score = torch.matmul(attn_weights, value_states)

                score = score.transpose(1, 2).contiguous()  # [B, seq, attn_head, head_dim]

                # replace after preparing data for next layer correctly
                self.check_check_target_module_exists(Dalora_config, adapter_name, model, submodule=attention.q_proj, inputs=XRMS)
                self.check_check_target_module_exists(Dalora_config, adapter_name, model, submodule=attention.k_proj, inputs=XRMS)
                self.check_check_target_module_exists(Dalora_config, adapter_name, model, submodule=attention.v_proj, inputs=XRMS)

                del attn_weights, query_states, key_states, value_states, XRMS

                # O matrix
                score = score.reshape(batch, seq, -1)  # [B, seq, attn_head * head_dim]
                attn_output = attention.o_proj(score)  # attention output

                self.check_check_target_module_exists(Dalora_config, adapter_name, model, submodule=attention.o_proj,
                                                      inputs=score)

                ''' MLP '''
                # prepare data for next layer first
                MLP = Decoder.mlp
                mlp_input = Decoder.post_attention_layernorm(inputs_embeds + attn_output)

                # replace after preparing data for next layer correctly
                self.check_check_target_module_exists(Dalora_config, adapter_name, model, submodule=MLP.up_proj,
                                                      inputs=mlp_input)

                self.check_check_target_module_exists(Dalora_config, adapter_name, model, submodule=MLP.gate_proj,
                                                      inputs=mlp_input)

                down_input = MLP.act_fn(MLP.gate_proj(mlp_input)) * MLP.up_proj(mlp_input)
                del mlp_input

                self.check_check_target_module_exists(Dalora_config, adapter_name, model, submodule=MLP.down_proj,
                                                      inputs=down_input)
                del down_input

                # test
                # 当前层的输出
                nadapters_embeds = Decoder(
                    hidden_states=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings
                )[0]
                nadapters_embeds = torch.nan_to_num(nadapters_embeds, nan=0.0)

                # get next Decoder's input
                inputs_embeds = nexts_embeds
                
        '''<<< 上面代码需要根据模型类型进行改写 >>>'''

        # 设置当前适配器为可训练
        self.set_adapter(self.active_adapters)  # 仅设置当前名称的adapter为可训练
        self._mark_only_adapters_as_trainable(model)  # 保证所有的adapter可训练

        # 推理模式
        if self.Dalora_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                # adapter_name可以是lora，被当前的模块名称包含
                if adapter_name in n:
                    p.requires_grad = False

    # 检测当前模块是否存在于目标模块列表中，如果存在则创建DaLoRA层
    def check_check_target_module_exists(
        self,
        Dalora_config,
        adapter_name,
        model,
        submodule: nn.Module,
        inputs: torch.tensor
    ):
        '''
            Args:
                :param submodule: current submodule to add adapter to
                :param inputs: sample data inputs for module
        '''

        key = get_name(model, submodule) # get module's name
        result = self._check_target_module_exists(Dalora_config, key) # 检查当前模块是否存在并且是目标模块

        if isinstance(result, _ExcludedModule):
            raise ValueError(
                "All modules were excluded. This is likely unintended. "
                "Check your `target_modules` and `exclude_modules` configuration."
            )
        elif not result:
            return
        else:
            # 模块存在
            self.targeted_module_names.append(key)
            parent, target, target_name = _get_submodules(model, key)

            print(f"---------Module name:{key}---------")
            
            # 打开文件并追加内容
            with open("/root/autodl-tmp/code/result.txt", "a", encoding="utf-8") as file:
                file.write(f"---------Module name:{key}---------\n")

            # 为目标模块创建adapter并替换
            self._create_and_replace(Dalora_config, adapter_name, target, target_name, parent, inputs, current_key=key)

    @staticmethod
    def _check_target_module_exists(DaLora_config, key):
        return check_target_module_exists(DaLora_config, key)

    # 创建DaLora层并替换掉原始模块
    def _create_and_replace(
        self,
        Dalora_config,
        adapter_name,
        target,
        target_name,
        parent,
        inputs,
        current_key,
    ):
        '''
            Args:
                :param target: 目标模块。
                :param target_name: 目标模块名称（single name，不是点式名称）。
                :param parent: 目标模块的直接父模块。
                :param inputs: sampling date as crrent module's inputs
                :param current_key: 当前模块的完整名称（点式名称）。
        '''

        # target_name_key保存pattern_keys中第一个与current_key匹配的键，即如果当前模块需要特殊Lora参数则将会被提取出
        pattern_keys = list(chain(Dalora_config.rank_pattern.keys(), Dalora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)

        # 获取当前层的Lora参数，如果没有找到则使用默认参数
        r = Dalora_config.rank_pattern.get(target_name_key, Dalora_config.r)
        alpha = Dalora_config.alpha_pattern.get(target_name_key, Dalora_config.Dalora_alpha)

        # 初始化参数
        kwargs = {
            "r": r,
            "Dalora_alpha": alpha,
            "Dalora_dropout": Dalora_config.Dalora_dropout,
        }

        if isinstance(target, DaLoraLayer):
            # 更新
            target.update_layer(
                adapter_name,
                r,
                inputs=inputs,
                Dalora_alpha=alpha,
                Dalora_dropout=Dalora_config.Dalora_dropout,
            )
        else:
            # 创建新的层
            new_module = self._create_new_module(adapter_name, inputs, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)

            # 用LoraLayer子类对象替换模型目标模块
            self._replace_module(parent, target_name, new_module, target)

    # 创建新的Dalora模块
    @staticmethod
    def _create_new_module(adapter_name, inputs, target, **kwargs):
        '''
            :param Dalora_config: config
            :param adapter_name: current adapter name
            :param inputs: sampling data as target's inputs
            :param target: module which needs to create adapter for
        '''
        return Linear(target, inputs, adapter_name, **kwargs)

    # 替换原始模型中的模块
    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)

        # 替换一个Dalora模块
        if hasattr(child, "base_layer"):
            child = child.base_layer

        meta = torch.device("meta")

        # 将新模块的子模块加载到模型所在的设备上
        for name, module in new_module.named_modules():
            if self.prefix in name:
                weight = next(child.parameters())
                if not any(p.device == meta for p in module.parameters()):
                    module.to(weight.device)

    # 设置指定的adapter为可训练
    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """
            Set the active adapter(s).
            Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True).
            Args:
                adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, DaLoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    # 仅将adapter置为可训练，同时根据bias选型控制bias参数是否可训练
    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        # 将没有Dalora标志的模块梯度置为不可训练
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        # 是否对bias的进行训练
        for active_adapter in self.active_adapters:
            bias = self.Dalora_config[active_adapter].bias
            if bias == "none":
                continue
            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in model.modules():
                    if isinstance(m, DaLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")