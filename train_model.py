import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

from Dataset.DataManager import DataManager
from Trainer.trainer import ModelTrainer
from Model_new.DaloraModel import DaloraLinear

def ChooseModel(method, model_path):
    '''
    :param method: fine-tune method
    :param model_path: model path
    :return: fine-tune model
    '''

    ''' Train LoRA model '''
    if method == 'lora':
        # 构建LoRA模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
        )
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            init_lora_weights=True,  # script_args.init_lora_weights,
            target_modules=[
                "q_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        return model

    ''' Train full fine-tune model '''
    if method == 'full':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        return model

    ''' Train DaLoRA model '''
    if method == 'dalora':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )

        for n, p in model.named_parameters():
            if "Dalora" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
                
        return model

''' choose fine-tune model '''
# tune_model = ChooseModel(method='lora', model_path='/root/autodl-fs/llama/LLM-Research/Meta-Llama-3-8B-Instruct')
tune_model = ChooseModel(method='dalora', model_path='/root/autodl-fs/model')

''' Build dataset & data collator '''
model_max_length = 512  # Maximum sequence length. Sequences will be right padded (and possibly truncated).
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='/root/autodl-fs/llama/LLM-Research/Meta-Llama-3-8B-Instruct',
    model_max_length=model_max_length,
    padding_side="right",
    use_fast=True,
    trust_remote_code=True
)

# 创建数据管理器
'''
# math:
# --data_path meta-math/MetaMathQA 
# --dataset_field query response 

# code:
#--data_path m-a-p/CodeFeedback-Filtered-Instruction 
#--dataset_field query answer

# Inst:
#--data_path fxmeng/WizardLM_evol_instruct_V2_143k 
#--dataset_field human assistant
'''
datapath = 'meta-math/MetaMathQA'
cache_dir = 'autodl-fs/dataset'
dataset_field = ['query', 'response']  # according to the dataset's key
datamanager = DataManager(
    tokenizer=tokenizer,
    cache_dir=cache_dir,
    datapath=datapath,
    dataset_split="train[:100000]",
    dataset_field=dataset_field
)

# 处理好的训练数据和包装后的含有data collator的字典
train_dataset = datamanager.mapping()
TrainData = datamanager.BuildCollactor(train_dataset)

''' Build Model Trainer '''
# 创建训练器
output_dir = '/root/autodl-fs/trained'
trainer = ModelTrainer(
    model=tune_model,
    tokenizer=tokenizer,
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=128,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=1,
    learning_rate=2e-5,
    weight_decay=0,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,
    tf32=True,
    report_to=None,
    isDalora=False,
)

# 训练并保存模型
trainer.train(TrainData)

# # train_model.py