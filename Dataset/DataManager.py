import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers import Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel

# ignore label
IGNORE_INDEX = -100

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """
    Tokenize a list of text strings.
    """
    # [Batch, Seqlen]
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    # 初始化label与inputs index相同
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]

    # 计算有效长度
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

# mapping函数
def train_tokenize_function(examples, tokenizer, query, response):
    # 输入输出
    sources = [
        PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]
    ]
    targets = [
        f"{output}{tokenizer.eos_token}" for output in examples[response]
    ]

    # 编码
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [tokenize_fn(strings, tokenizer) for strings in (examples, sources)]

    # 构造label
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)

# 数据collator
@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # [Batch, Seqlen]
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        # 张量化输入
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # 张量化label
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

# 数据管理类
class DataManager:
    def __init__(
        self,
        tokenizer,
        datapath,
        dataset_split,
        dataset_field
    ):
        '''
        Args:
        :param tokenizer: model's tokenizer
        :param datapath: dataset's path
        :param dataset_split: dataset split number
        :param dataset_field: datasets keys field
        '''
        self.tokenizer = tokenizer
        self.datapath = datapath
        self.dataset_split = dataset_split
        self.dataset_field = dataset_field  # 数据字段

    # 将数据进行padding并且配置对应label，对于需要忽略的输入prompt token对应的label，置为默认的-100
    def mapping(self):
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # 设置padding token
        raw_train_datasets = load_dataset(self.data_path, split=self.dataset_split)  # 加载数据

        # 构建训练格式的数据，数据类型不是张量
        train_dataset = raw_train_datasets.map(
            train_tokenize_function,
            batched=True,
            batch_size=3000,
            num_proc=16,  # 32
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",

            # 除了example参数（默认）外的其余参数需要手动添加
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "query": self.dataset_field[0],
                "response": self.dataset_field[1]
            }
        )

        return train_dataset
    # 构建数据collator，用于进行训练
    def BuildCollactor(self, train_dataset):
        data_collator = DataCollator(tokenizer=self.tokenizer)
        return dict(train_dataset=train_dataset, data_collator=data_collator)
