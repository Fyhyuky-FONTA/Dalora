import copy
import os
from transformers import Trainer, TrainingArguments
import torch
import json

# 训练器
class ModelTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        output_dir,
        num_train_epochs,
        per_device_train_batch_size,
        gradient_accumulation_steps,
        save_strategy,
        save_steps,
        save_total_limit,
        learning_rate,
        weight_decay,
        warmup_ratio,
        lr_scheduler_type,
        logging_steps,
        bf16,
        tf32,
        report_to,
        isDalora
    ):
        '''
        Args:
        :param model: model to be trained
        :param tokenizer: model's tokenizer
        :param output_dir: path to save the trained model
        :param num_train_epochs: number of epoch to train model
        :param per_device_train_batch_size: the batch size per device (GPU/TPU) for training
        :param gradient_accumulation_steps: The number of update steps to accumulate before
            performing a backward/update pass
        :param save_strategy: The strategy to save the model during training. Options include
            "no", "epoch", or "steps", determining when to save the model checkpoints
        :param save_steps: The number of steps between each save of the model, applicable when
            `save_strategy` is set to "steps"
        :param save_total_limit: The maximum number of checkpoints to keep. If exceeded, older
            checkpoints will be deleted
        :param learning_rate:The initial learning rate for the optimizer. This controls how much
            to change the model in response to the estimated error each time the model weights are updated
        :param weight_decay: The weight decay to apply (if any). This is a regularization technique
            that helps prevent overfitting by penalizing large weights
        :param warmup_ratio: The ratio of total steps to perform learning rate warmup
        :param lr_scheduler_type: The type of learning rate scheduler to use. Options may include
            "linear", "cosine", "cosine_with_restarts", etc., which define how the learning rate
            changes during training
        :param logging_steps: The number of steps between each log output
        :param bf16: Whether to use bfloat16 (bf16) precision for training
        :param tf32: Whether to use TensorFloat32 (tf32) precision for training. This is particularly
            useful on NVIDIA A100 GPUs to improve performance
        :param report_to: The reporting backend to use for logging and tracking metrics. Options
            may include "tensorboard", "wandb", etc., depending on the logging framework you want to use
        '''
        self.model = model  # 待训练模型
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.parameters = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=logging_steps,
            bf16=bf16,
            tf32=tf32,
            report_to=report_to,
            logging_dir='/root/autodl-tmp/code/logs'
        )

        self.isDalora = isDalora

    # 获取模型的所有参数和所有可训练参数
    def get_nb_trainable_parameters(self):
        '''
        :return: the number of trainable parameters and the number of all parameters in the model.
        '''
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_bytes = param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def train(self, TrainData):
        '''
        :param TrainData: Dict of {train_dataset:train dataset Dict, data_collator:DataCollator Object}
        '''
        # 显示梯度
        for n, p in self.model.named_parameters():
            print(n, p.requires_grad)

        # 显示训练参数数量
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

        # 创建训练器
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.parameters,
            **TrainData
        )

        # 训练
        self.model.config.use_cache = False
        trainer.train()

        # 测试
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 准备问答函数
        def answer_question(question):
            inputs = self.tokenizer(question, return_tensors="pt").to(device)
            outputs = self.model.generate(**inputs, max_length=100)  # 设置生成的最大长度
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer

        # 示例问答
        question = "什么是人工智能？"
        answer = answer_question(question)
        print("问题:", question)
        print("答案:", answer)

        trainer.save_state()

        # 保存
        if self.isDalora is False:
            # 合并adapter
            self.model = self.model.merge_and_unload()
            self.tokenizer.save_pretrained(self.output_dir)
            self.model.save_pretrained(self.output_dir)
        else:
            # 直接保存模型和分词器
            self.tokenizer.save_pretrained(self.output_dir)
            self.model.save_pretrained(self.output_dir)

            # 向原始的的config对象中添加额外的重要参数
            config = self.model.config.to_dict()

            config["auto_map"] = {
                "AutoConfig": "DaloraConfig.DaloraConfig",
                "AutoModelForCausalLM": "DaloraModel.DaloraLlamaForCausalLM",
            }
            config["architectures"] = ["DaloraLlamaForCausalLM"]

            # 两个定义文件复制到指定路径下
            os.system(
                "cp ./Model_new/DaloraConfig.py ./Model_new/DaloraModel.py "
                + self.output_dir
            )

            # 添加新的属性
            config["module_dict"] = list(self.model.state_dict().keys())
            config["extra_dict"] = {
                'default': {
                    "dropout": 0.00,
                    "rank": 32,
                    "scaling": 1
                }
            }

            # 保存config文件
            json.dump(config, open(self.output_dir + "/config.json", "w"), indent=2)