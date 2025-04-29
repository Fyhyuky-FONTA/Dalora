"""
This file is only used to separate the linear layer parameters and decomposition results from the model,
and save them in a different location. It is currently for testing purposes only.
"""
import torch
import numpy as np
from typing import List, Optional
import os
from FastSVD.FastSVD import FastSVD

# 构建各层的适配器
@torch.no_grad()
class AdapterConstructer:
    'construct adapter parameters B & A for each decoder layer in Llama model under the experiment'
    def __init__(self, model, tokenizer, sample_X: List[str], path):
        '''
        Args:
            :param model: LLM which to bulid 'Adapter' on, here's to be Llama
            :param tokenizer: LLM tokenizer for encoding
            :param sample_X: List[str], data samples, shape of [BL, din], where din = 4096 under Llama model,
                meanwhile din = dout
            :param path: directory to save tensor
        '''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = path
        self.tokenizer = tokenizer  # 分词器
        self.model = model  # 模型
        self.data = sample_X  # 数据采样点
        # self.Building(path) # 获取Llama每一层的q，k，v与参数计算得到的矩阵XW，将张量保存到指定文件路径

    # 构建并返回所有层的A，B矩阵
    def Building(self):
        self.tokenizer.add_special_tokens({'pad_token': '<|reserved_special_token_0|>'})
        X = self.tokenizer(self.data, return_tensors="pt", padding=True, truncation=True)  # 编码
        input_ids = X["input_ids"].to(self.device)  # 需要将类型转化为long

        # 过Embedding层
        EmbedLayer = self.model.model.embed_tokens  # Embedding Layer
        inputs_embeds = EmbedLayer(input_ids)  # [B, L, din]，数据类型为bfoat16
        batch, seq, _ = inputs_embeds.shape

        print("Embedding tensor shape:", inputs_embeds.shape)

        # 创建mask矩阵（每个注意力头分配一组，数据类型bfoat16
        attention_mask = X["attention_mask"].to(self.device)
        dtype, device = inputs_embeds.dtype, inputs_embeds.device  # 获取Embedding张量的信息
        target_length = attention_mask.shape[-1]
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)  # cache位置参数
        attention_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=seq,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=inputs_embeds.shape[0]
        )
        print("Attention mask shape:", attention_mask.shape)

        # 使用虚张量获取旋转位置编码张量，对照源码进行编写
        position_ids = cache_position.unsqueeze(0)
        lama_model = self.model.model
        attn = lama_model.layers[0].self_attn  # 获取第一层的attention层即可
        virtual = torch.rand(batch, seq, attn.num_key_value_heads, attn.head_dim).transpose(1, 2).to(dtype)  # 注意需要进行类型转换
        position_embeddings = attn.rotary_emb(virtual, position_ids)

        # 将张量过每一层进行
        for layer_idx in range(len(lama_model.layers)):
            Decoder = lama_model.layers[layer_idx]
            XRMS = Decoder.input_layernorm(inputs_embeds)  # RMS归一化层，是attention层的输入

            print("RMS Layer output shape:", XRMS.shape)

            # 保存张量到文件，模型的attention没有bias
            torch.save(XRMS, self.path + f'/layer_{layer_idx}_X.pt')
            torch.save(Decoder.self_attn.q_proj.weight.data, self.path + f'/layer_{layer_idx}_Q.pt')
            torch.save(Decoder.self_attn.k_proj.weight.data, self.path + f'/layer_{layer_idx}_K.pt')
            torch.save(Decoder.self_attn.v_proj.weight.data, self.path + f'/layer_{layer_idx}_V.pt')
            print(f"Layer {layer_idx} saved")

            del XRMS

            # 不返回attention输出和Cache输出模式
            inputs_embeds = Decoder(
                hidden_states=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )[0]
            print(f"Decoder Layer{layer_idx} output shape:", inputs_embeds.shape)

    # 创建4维张量，使用的是Llama源码中的mask构造方式
    def _prepare_4d_causal_attention_mask_with_cache_position(
            self,
            attention_mask: torch.Tensor,
            sequence_length: int,
            target_length: int,
            dtype: torch.dtype,
            device: torch.device,
            cache_position: torch.Tensor,
            batch_size: int,
    ):
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask

        # 创建mask张量矩阵
        else:
            min_dtype = torch.finfo(dtype).min  # 获取当前数值类型下的最小值

            # 初始化形状为[seq, cache_seq + seq]的最小值矩阵
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )

            # 主对角线上方填充causal_mask对应位置的值
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0

                # 将padding_mask中True对应在causal_mask中的相同位置的元素替换为最小值，处理padding token的掩码
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask,
                                                                                                    min_dtype)
        return causal_mask

    # 通过梯度学习或者直接计算出每一层参数的最优分解式
    def AdapterWeight(
            self,
            path_to_save,
            epoch:int,
            rank:int = 8,
            batch_size:int = 1024,
            learning_rate:float = 0.001,
            regularization=None,
            lamb=None,
            cross_regu=None,
            method:Optional[str]='train'
    ):
        '''
        Args:
            :param path_to_save: path to save B & A matrix of each layer for Wq, Wk, Wv
            :param epoch: training epoch
            :param rank: rank limitation
            :param batch_size: training batch size
            :param learning_rate: learning rate
            :param regularization: int, number of U's columns to sample for standardization of column close to 1
            :param lamb: float, lambda for regularization
            :param cross_regu: int, number of U's columns to sample for column vector orthogonalization
            :param method: Optional['compute', 'train'], default to 'train', which means learning U matrix with SGD,
                'compute' means to obtain U directly
        '''

        # 遍历所有文件
        lama_model = self.model.model
        for layer_idx in range(len(lama_model.layers)):
            x_file = os.path.join(self.path, f'layer_{layer_idx}_X.pt')
            q_file = os.path.join(self.path, f'layer_{layer_idx}_Q.pt')
            k_file = os.path.join(self.path, f'layer_{layer_idx}_K.pt')
            v_file = os.path.join(self.path, f'layer_{layer_idx}_V.pt')

            X = torch.load(file).view(-1, X.shape[-1]) # [B, L, dim] -> [B * L, dim]
            dtype = X.dtype

            # 类型转化为float64
            if X.dtype != torch.float64:
                X = X.to(torch.float64)

            X = X.cpu().numpy() # 转化为numpy

            # 分别处理Q，K，V参数矩阵
            for mat, file in zip(('Q', 'K', 'V'), (q_file, k_file, v_file)):
                W = torch.load(file) # [dim, head_nums * head_dim / KV_heads * head_dim]
                if W.dtype != torch.float64:
                    W = W.to(torch.float64)
                W = W.cpu().numpy()

                # 创建快速SVD对象
                fast = FastSVD(
                    X=X,
                    W=W,
                    r=rank,
                    batch_size=batch_size,
                    lr=learning_rate,
                    regu=regularization,
                    lamb=lamb,
                    cross=cross_regu
                )

                # 训练还是直接计算出闭式解
                if method == 'train':
                    fast.train(epoch)
                    U = fast.U.data, S = fast.Sr, VT = fast.VTr
                else:
                    U, S, VT = fast.ComputeU()
                    dig = [i for i in range(rank)]
                    S = S[:rank, :rank], U = U[:, :rank], VT = VT[:rank, :]

                # 得到B & A
                B = np.linalg.inv(X @ X.T) @ X.T @ U @ np.sqrt(S)
                A = np.sqrt(S) @ VT

                # 转化为特定类型的tensor保存
                torch.save(torch.tensor(B, dtype=dtype), path_to_save + f'/layer_{layer_idx}_{mat}_B.pt')
                torch.save(torch.tensor(A, dtype=dtype), path_to_save + f'/layer_{layer_idx}_{mat}_A.pt')
                print(f"Layer {layer_idx} B & A saved")