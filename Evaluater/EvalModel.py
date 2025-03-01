import torch
from lm_eval.base import BaseLM

# 测试模型
class EvalModel(BaseLM):
    def __init__(
        self,
        model,
        tokenizer,
        batch_size=1,
    ):
        super().__init__()
        assert isinstance(batch_size, int)
        self._device = model.device

        self.model = model
        self.model.eval()  # 将基模型置为测试模式

        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size
        self.seqlen = 2048

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            return self.model.config.max_position_embeddings  # gptneoconfig doesn't have n_ctx apparently

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    # 编码解码
    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    # 输出logits张量，这里需要和词汇表进行同步
    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence], the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :]

    # 模型生成回答
    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
