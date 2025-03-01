import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from Evaluater.evaluater import Evaluater

# load model
model_dir = "/root/autodl-tmp/llama/LLM-Research/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")

# evaluate model
evalu = Evaluater(
    model=model,
    tokenizer=tokenizer,
    model_name='Llama',
    tasks="mmlu",
    datasets="wikitext2",
    limit=-1,
)

evalu.evaluate()

# evaluate_model.py