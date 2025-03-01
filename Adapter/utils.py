import re
import torch

class _ExcludedModule:
    """
    A private helper method used to represent excluded modules in the check_target_module_exists function.
    """
    def __bool__(self):
        return False

# 检查模块是否存在
def check_target_module_exists(config, key: str) -> bool | re.Match[str] | None:
    """
    A helper method to check if the passed module's key name matches any of the target modules in the adapter_config.
    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match target modules from
        key (`str`): A key to search any matches in config
    Returns:
        `bool` | `re.Match[str]` | `None`: True of match object if key matches any target modules from config, False or
        None if no match found
    """
    if hasattr(config, "exclude_modules") and config.exclude_modules:
        if isinstance(config.exclude_modules, str):
            if re.fullmatch(config.exclude_modules, key):
                return _ExcludedModule()
        elif key in config.exclude_modules:
            return _ExcludedModule()
        elif any(key.endswith(f".{exclude_key}") for exclude_key in config.exclude_modules):
            return _ExcludedModule()

    if isinstance(config.target_modules, str):
        target_module_found = re.fullmatch(config.target_modules, key)
    elif key in config.target_modules:
        # this module is specified directly in target_modules
        target_module_found = True
    else:
        target_module_found = any(key.endswith(f".{target_key}") for target_key in config.target_modules)

        layer_indexes = getattr(config, "layers_to_transform", None)
        layers_pattern = getattr(config, "layers_pattern", None)

        is_using_layer_indexes = layer_indexes is not None and (
            len(layer_indexes) != 0 if isinstance(layer_indexes, list) else True
        )
        if is_using_layer_indexes and target_module_found:
            layer_index = None
            # TODO: It's still unclear how empty layers_pattern (None, [], or "") should behave
            # For now, empty layers_pattern means any layer pattern is ok
            if layers_pattern is None or len(layers_pattern) == 0:
                layer_index = re.match(r".*\.[^.]*\.(\d+)\.", key)
            else:
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
                for pattern in layers_pattern:
                    layer_index = re.match(rf".*\.{pattern}\.(\d+)\.", key)
                    if layer_index is not None:
                        break

            if layer_index is None:
                target_module_found = False
            else:
                layer_index = int(layer_index.group(1))
                if isinstance(layer_indexes, int):
                    target_module_found = layer_index == layer_indexes
                else:
                    target_module_found = layer_index in layer_indexes

    return target_module_found

# 根据模块名称获取一个模块的父模块、模块本身和模块本身名称
def _get_submodules(model, key):
    """
    根据模块名称获取一个模块的父模块、模块本身和模块本身名称
    case in jupyter lab:
    >>> from modelscope import Model

    >>> cache_dir = "E:\\Tuna\\Qwen2-0.5B"
    >>> model = Model.from_pretrained('qwen\\Qwen2-0.5B', cache_dir=cache_dir)

    >>> path = 'C:\\Users\\86152\\.cache\\modelscope\\hub\\qwen\\Qwen2-0___5B'
    >>> config = AutoConfig.from_pretrained(path)
    >>> og_model = AutoModelForCausalLM.from_config(config)

    >>> _get_submodules(og_model, 'model.layers.0.self_attn.q_proj')

    out:
        (Qwen2SdpaAttention(
           (q_proj): Linear(in_features=896, out_features=896, bias=True)
           (k_proj): Linear(in_features=896, out_features=128, bias=True)
           (v_proj): Linear(in_features=896, out_features=128, bias=True)
           (o_proj): Linear(in_features=896, out_features=896, bias=False)
           (rotary_emb): Qwen2RotaryEmbedding()
         ),
         Linear(in_features=896, out_features=896, bias=True),
         'q_proj')
    """

    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

# 创建4维张量，使用的是Llama源码中的mask构造方式
def _prepare_4d_causal_attention_mask_with_cache_position(
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

# 为输入的张量最后一维进行交替编码，x通常为[B, seq, dim]
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# 添加位置编码
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 这个函数将每一层的K和V进行重复性扩展，完成后序GQA的注意力计算，将[B, KV_head, seq, head_dim]按照第二维复制为[B, attn_head, seq, dim]形状的张量
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape # [B, KV_head, seq, head_dim]
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# 获取当前模块的完整名称
def get_name(model, module):
    '''
    :param module: nn.Module
    :return: str, current module's name
    '''
    for name, mod in model.named_modules():
        if mod is module:
            return name

# 获取当前模块的完整名称
def get_module(model, name):
    '''
    :param name: str module name
    :return: nn.Module, current module
    '''
    for na, module in model.named_modules():
        if na == name:
            return module