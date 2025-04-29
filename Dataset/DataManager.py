import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers import Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel

# ignore label to indicate the position in which the tokens don't need to compute loss
IGNORE_INDEX = -100

# prompt template prepared for instructions
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """
        Tokenize a list of text strings.
        Args:
            strings: List of str to tokenize.
            tokenizer: model's tokenizer.
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

    # init label and inputs index
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]

    # calculate the effective length using `string.ne`
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def train_tokenize_function(examples, tokenizer, query, response):
    """
        Mapping function to map a list of strings to a list of int.
        Args:
            examples: a batch of data with.
            tokenizer: model's tokenizer.
            query: a field name for indexing instruction data in examples.
            response: a field name for indexing answer data in examples.
    """
    # input and output
    sources = [
        PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]
    ]
    targets = [
        f"{output}{tokenizer.eos_token}" for output in examples[response]
    ]

    # encoding List of str
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [tokenize_fn(strings, tokenizer) for strings in (examples, sources)]

    # building label
    input_ids = examples_tokenized["input_ids"]  # List of int
    labels = copy.deepcopy(input_ids)

    # using IGNORE_INDEX to mark padding token
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # [Batch, Seqlen]
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        # tensorize input
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # tensorize label
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

class DataManager:
    """A class for data management."""
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
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # setting padding token
        raw_train_datasets = load_dataset(self.data_path, split=self.dataset_split)  # load data

        '''
            Construct training format data, where the data type is not a tensor.
            All extra parameters except the `example` parameter (default) in the mapping function 
            `train_tokenize_function` need to be added manually to call the mapping function 
            train_tokenize_function.
        '''
        #
        train_dataset = raw_train_datasets.map(
            train_tokenize_function,
            batched=True,
            batch_size=3000,
            num_proc=16,  # 32
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
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
