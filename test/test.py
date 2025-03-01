from pprint import pprint
import torch
from Adapter.model import DaLoraBuilder
from Adapter.config import DaLoraConfig
from Dataset.datautils import DataSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

# 加载模型
model_dir = r"/root/autodl-tmp/llama/LLM-Research/Meta-Llama-3-8B-Instruct"  
tokenizer = AutoTokenizer.from_pretrained(model_dir)  

# 读取保存的模型
save_directory = "/root/autodl-tmp/code/model"
with open(os.path.join(save_directory, "model.pkl"), "rb") as model_file:
    model = pickle.load(model_file)