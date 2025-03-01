# import torch
# from pprint import pprint
# import pickle
# import os
#
# from Model.DaloraModel import DaloraLlamaForCausalLM
# from transformers import AutoConfig
#
# # 加载整个模型的参数字典并获取键值
# model_path = r'/root/autodl-tmp/code/model/model_with_grad_info.pth'
# dalora_model = torch.load(model_path, weights_only=False)
#
# # 加载配置类
# model_dir = "/root/autodl-tmp/llama/LLM-Research/Meta-Llama-3-8B-Instruct"
# config = AutoConfig.from_pretrained(model_dir)
#
# # 按照格式创建额外字典
# extra_dict = {
#     'default': {
#         "dropout": 0.0,
#         "rank": 8,
#         "scaling": 32
#     }
# }
#
# # 重构
# model = DaloraLlamaForCausalLM(
#     config=config,
#     module_dict=dalora_model['model_state_dict'],
#     extra_dict=extra_dict,
# )
#
# pprint(model)
#
# # 加载模型参数
# model.load_state_dict(dalora_model['model_state_dict'])
#
# # 恢复requires_grad信息
# for name, param in model.named_parameters():
#     param.requires_grad = dalora_model['requires_grad'][name]
#
# # # 查看梯度
# # for n, p in model.named_parameters():
# #     print(n, p.requires_grad)
#
# # 将模型按照hugging face格式进行保存
# # rebuild_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from Model_new.DaloraModel import DaloraLinear
from pprint import pprint

import torch

# 加载模型
# model_dir = r"/root/autodl-fs/llama/LLM-Research/Meta-Llama-3-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForCausalLM.from_pretrained(
#     model_dir, 
#     torch_dtype=torch.bfloat16, 
#     device_map="auto"
# )

# 模型线性层中没有Nan值
model_name_or_path = '/root/autodl-fs/model'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# # 查看线性层的统计特性
# for n, p in model.named_parameters():
#     if 'weight' in n:  # 只检查线性层的权重
#         if torch.isnan(p).any():
#             print(f"Parameter '{n}' contains NaN values.")
#         else:
#             max_val = p.max().item()
#             min_val = p.min().item()
#             print(f"Parameter '{n}' - Max: {max_val}, Min: {min_val}")

# # 设置梯度
# for n, p in model.named_parameters():
#     print(n)
#     if not isinstance(p, DaloraLinear):
#         p.requires_grad = False

pprint(model)

# 测试
device = "cuda" if torch.cuda.is_available() else "cpu"

# 准备问答函数
def answer_question(question):
    inputs = tokenizer(question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=100)  # 设置生成的最大长度
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 示例问答
question = "什么是人工智能，使用中文回答？"
answer = answer_question(question)
print("问题:", question)
print("答案:", answer)

# rebuild_model.py