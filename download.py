from modelscope import snapshot_download, AutoModel, AutoTokenizer

model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='/home/zps/DaLoRA/Llama', revision='master')
print(model_dir)  # /home/zps/DaLoRA/Llama/LLM-Research/Meta-Llama-3-8B-Instruct