from datasets import load_dataset
import random
import json
import io
import os

from tqdm import tqdm

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

# 对模型的输入数据进行采样
def DataSampler(
    name,
    subname,
    field,
    tokenizer,
    nsamples,
    seqlen,
    isconcate=False,
):
    '''
    This function provide method to construct sampling data for DaLoRA tuning, we can use text-only style datasets 
    or Q & A datasets for sampling process, but we should concatenate some text-only style datasets for more complete
    content while sampling, for Q & A datasets we can use point sampling orienting single data.
    Args:
    :param name: name of the chosen dataset
    :param subname: sub dataset name of the chosen dataset
    :param field: dataset's key, 'text' for '"wikitext"' dataset, 'sentence' for 'ptd' dataset, 'question' 
        for 'nq_open' dataset
    :param tokenizer: model's tokenizer
    :param nsamples: number of sampling data
    :param seqlen: set max sequence length if we choose to concatenate all text data
    :param isconcate: for Q & A style datasets, set `False` to avoid data concatenate while sampling
    '''
    cache_dir = f"dataset/{name}"

    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    # 检测cache
    if os.path.exists(cache_dir):
        print(f"found data file: {cache_dir}")

        # 读取特定子任务
        data = load_dataset(
            path=name,
            name=subname,
            cache_dir=cache_dir
        )
    else:
        # 选择不同数据集进行实验
        if name == "wikitext2":
            # 纯文本数据集，字段名称只有text：https://huggingface.co/datasets/Salesforce/wikitext
            data = load_dataset(
                path="wikitext",
                name="wikitext-2-raw-v1",
                cache_dir=cache_dir
            )
        elif name == "ptb":
            # 纯文本数据集，字段名称只有sentence：https://huggingface.co/datasets/ptb-text-only/ptb_text_only
            data = load_dataset(
                path="ptb_text_only",
                name="penn_treebank",
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        elif name == "google-research-datasets/nq_open":
            # 问答数据集
            data = load_dataset(
                path="google-research-datasets/nq_open",
                cache_dir=cache_dir
            )
        else:
            raise NotImplementedError

    random.seed(233)

    traindata = data["train"]

    # 选择concate
    traindataset = []
    if isconcate:
        tot_text = "\n\n".join(traindata[field])
        for _ in range(nsamples):
            # 采样编码后按照长度截断
            i = random.randint(0, len(tot_text) - seqlen - 1)
            j = i + seqlen * 10
            trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
            inp = trainenc.input_ids[:, :seqlen]

            # 转化为纯文本
            token_ids = inp[0].tolist()
            traindataset.append(tokenizer.decode(token_ids, skip_special_tokens=True))

    # 单点采样并按照格式包装为prompt
    else:
        # 注意我们需要使用left padding进行padding，因为我们使用的是生成模型
        for _ in tqdm(range(nsamples)):
            # 采样数据
            i = random.randint(0, len(traindata))
            data = traindata[field][i]

            # 添加模板
            messages = [{"role": "user", "content": data}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            traindataset.append(text)

    return traindataset

# 加载测试数据集
def get_eval_loaders(name, tokenizer):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    if "ptb" in name:
        valdata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    if "c4" in name:
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    raise NotImplementedError