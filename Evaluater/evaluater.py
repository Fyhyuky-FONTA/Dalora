import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import load_from_disk
from lm_eval import evaluator
from Evaluater.EvalModel import EvalModel
from Evaluater.evalutils import (
    MMLUTASKS,
    QATASKS
)

# 测评对象
class Evaluater:
    def __init__(
        self,
        model,
        tokenizer,
        model_name,
        tasks,
        datasets="",
        num_fewshot=0,
        limit=-1,
        batch_size=1,
    ):
        '''
        Args:
        :param model: model to be evaluated
        :param tokenizer: model's tokenizer
        :param model_name: current model name
        :param tasks: str Optional ["mmlu",  "llmqat"], tasks option
        :param datasets: str, multi-datasets names separate by ',' which to, like "wikitext2,ptb"
        :param num_fewshot: Number of examples in few-shot context
        :param limit: number of test samples for debug, set to -1 is no limit
        :param batch_size: size of a batch while evaluating model
        '''
        self.model = EvalModel(model, tokenizer, batch_size=batch_size)
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.tasks = tasks
        self.datasets = datasets
        self.num_fewshot = num_fewshot
        self.limit = limit
        self.batch_size = batch_size

    # 测试模型能力
    def evaluate(self):
        Evalmodel = self.model
        results = {}
        if self.datasets:
            # 取出当前数据集的名称
            for dataset in self.datasets.split(","):
                '''
                Load dataset, all datasets should be download into file 'dataset'
                '''
                testdata = load_from_disk(f"/root/autodl-tmp/code/Dataset/{dataset}")
                textloader = self.tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
                testenc = textloader['input_ids']

                nsamples = testenc.numel() // Evalmodel.seqlen  # 元素总数除以模型输入长度

                # 保存当前cache选项
                use_cache = Evalmodel.model.config.use_cache
                Evalmodel.model.config.use_cache = False

                # 评估
                Evalmodel.model.eval()
                nlls = []

                # 添加此行以避免计算图的创建，不然会引起内存累积
                with torch.no_grad():
                    for i in tqdm(range(nsamples)):
                        # 获取模型输出
                        batch = testenc[:, (i * Evalmodel.seqlen): ((i + 1) * Evalmodel.seqlen)].to(Evalmodel.device)
                        outputs = Evalmodel.model.model(batch)

                        # 得到预测输出
                        hidden_states = outputs[0]
                        logits = Evalmodel.model.lm_head(hidden_states)  # .contiguous()

                        # 移位
                        shift_logits = logits[:, :-1, :]  # .contiguous()
                        shift_labels = testenc[:, (i * Evalmodel.seqlen): ((i + 1) * Evalmodel.seqlen)][:, 1:].to(
                            Evalmodel.device)

                        # 计算损失
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),  # .to(shift_labels.device),
                            shift_labels.view(-1).to(shift_logits.device),
                        )
                        neg_log_likelihood = loss.float() * Evalmodel.seqlen
                        nlls.append(neg_log_likelihood)

                        # 达到指定测试数量则停止
                        if i == self.limit:
                            break

                # 当前数据集的全局损失
                ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * Evalmodel.seqlen))
                print(dataset, ppl.item())

                # 重置cache选项
                Evalmodel.model.config.use_cache = use_cache

                results[dataset] = ppl.item()

        # 选择测试任务
        if self.tasks == "mmlu":
            tasks = MMLUTASKS
        if tasks == "llmqat":
            tasks = QATASKS

        # 进行测试
        if tasks != "":
            t_results = evaluator.simple_evaluate(
                Evalmodel,
                tasks=tasks,
                batch_size=self.batch_size,
                num_fewshot=self.num_fewshot,
                limit=None if self.limit == -1 else self.limit,
                no_cache=True,
            )
            t_results = t_results["results"]
            acc_list = [
                t_results[key]["acc"] for key in t_results.keys() if "acc" in t_results[key]
            ]
            t_results["mean"] = sum(acc_list) / len(acc_list)
            results.update(t_results)

            print(results)
            print(f"\n\n===== mean acc: {sum(acc_list) / len(acc_list)} =====\n\n")

        return results