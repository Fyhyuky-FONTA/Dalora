# '''
# 将数据经过模型的各个层，实现对于adapter参数的矩阵的构建，最终保存模型各个层的参数，
# 如果需要使用模型，需要在rebuild_model.py中重构并按照hugging face的格式进行保存，
# 这样才能够使用模型。
# '''
# from pprint import pprint
# import torch
# from Adapter.model import DaLoraBuilder
# from Adapter.config import DaLoraConfig
# from Dataset.datautils import DataSampler
# from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
#
# # 加载模型
# model_dir = r"/root/autodl-tmp/llama/LLM-Research/Meta-Llama-3-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")
#
# # 准备批量输入文本
# texts = DataSampler(
#     name="google-research-datasets/nq_open",
#     subname=None,
#     field='question',
#     tokenizer=tokenizer,
#     nsamples=100,
#     seqlen=None,
#     isconcate=False
# )
#
# # config对象
# config = DaLoraConfig(
#     task_type="CAUSAL_LM",
#     target_modules=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#     ],
#     inference_mode=False,
#     r=8,
#     Dalora_alpha=32,
#     Dalora_dropout=0.001,
# )
#
# # 创建模型
# dalora_model = DaLoraBuilder(tokenizer, model, texts, config, adapter_name='default').model
#
# # # 查看梯度
# # for n, p in dalora_model.named_parameters():
# #     print(n, p.requires_grad)
#
# # 保存模型参数和 requires_grad 信息
# state = {
#     'model_state_dict': dalora_model.state_dict(),
#     'requires_grad': {
#         name: param.requires_grad for name, param in dalora_model.named_parameters()
#     }
# }
# torch.save(state, r'model/model_with_grad_info.pth')
#
#
# # # 保存模型
# # torch.save(dalora_model.state_dict(), r'model/model_state_dict.pth')
#
# # build_dalora_model.py
# # 读取模型，并且进行初始的问答测试

'''
将数据经过模型的各个层，实现对于adapter参数的矩阵的构建，最终保存模型各个层的参数，
如果需要使用模型，需要在rebuild_model.py中重构并按照hugging face的格式进行保存，
这样才能够使用模型。
'''
from pprint import pprint
import json
import os
import torch
from Adapter.model import DaLoraBuilder
from Adapter.config import DaLoraConfig
from Dataset.datautils import DataSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

# 加载模型
model_dir = r"/root/autodl-fs/llama/LLM-Research/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 生成排除最后一个GPU的设备列表
num_gpus = torch.cuda.device_count()
if num_gpus > 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(num_gpus-1)))
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(num_gpus)))
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda:0"
    )

# # 数据并行
# model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

# 准备批量输入文本
texts = DataSampler(
    name="google-research-datasets/nq_open",
    subname=None,
    field='question',
    tokenizer=tokenizer,
    nsamples=500,
    seqlen=None,
    isconcate=False
)

# config对象
r = 8
config = DaLoraConfig(
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj"
    ],
    inference_mode=False,
    r=r,
    Dalora_alpha=32,
    Dalora_dropout=0.0,
)

# 创建模型
dalora_model = DaLoraBuilder(tokenizer, model, texts, config, adapter_name='default').model

# # 查看梯度
# for n, p in dalora_model.named_parameters():
#     print(n, p.requires_grad)

# 将模型按照hugging face格式保存
save_path = '/root/autodl-fs/model'

# 直接保存模型和分词器
tokenizer.save_pretrained(save_path)
dalora_model.save_pretrained(save_path)

print("Saved!")

# 向原始的的config对象中添加额外的重要参数
config = dalora_model.config.to_dict()

config["auto_map"] = {
    "AutoConfig": "DaloraConfig.DaloraConfig",
    "AutoModelForCausalLM": "DaloraModel.DaloraLlamaForCausalLM",
}
config["architectures"] = ["DaloraLlamaForCausalLM"]

# 两个定义文件复制到指定路径下
os.system(
    "cp ./Model_new/DaloraConfig.py ./Model_new/DaloraModel.py "
    + save_path
)

# 添加新的属性
config["module_dict"] = list(dalora_model.state_dict().keys())

print(type(dalora_model.state_dict().keys()))

config["extra_dict"] = {
    'default': {
        "dropout": 0.00,
        "rank": 8,
        "scaling": 4
    }
}

# 保存config文件
json.dump(config, open(save_path + "/config.json", "w"), indent=2)

# build_dalora_model.py
# 在运行的过程中，GPU的内存占用是逐步变大的，因为每一层都会被添加上新的adapter，
# 于是将会逐步使参数量上升
# 读取模型，并且进行初始的问答测试