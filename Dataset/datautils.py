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
        This function provide method to construct sampling data for DaLoRA tuning, we can use text-only style
        datasets or Q & A datasets for sampling process, but we should concatenate some text-only style datasets
        for more complete content while sampling, for Q & A datasets we can use point sampling orienting single
        data.
        Args:
        :param name: name of the chosen dataset.
        :param subname: sub dataset name of the chosen dataset.
        :param field: dataset's key, 'question' for 'nq_open' dataset, other dataset can be used in the same way.
        :param tokenizer: model's tokenizer.
        :param nsamples: number of sampling data.
        :param seqlen: set max sequence length if we choose to concatenate all text data.
        :param isconcate: for Q & A style datasets, set `False` to avoid data concatenate while sampling.
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
        if name == "google-research-datasets/nq_open":
            data = load_dataset(
                path="google-research-datasets/nq_open",
                cache_dir=cache_dir
            )
        else:
            raise NotImplementedError

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