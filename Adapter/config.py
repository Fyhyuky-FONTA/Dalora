import logging
import os

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft import PeftConfig

logger = logging.getLogger(__name__)

@dataclass
class DaLoraConfig(PeftConfig):
    '''
    Main Attributes：
    -target_modules：用于指定要替换为 DaLoRA 的模块名称或模块名称的正则表达式，示例：可以是一个字符串列表，如['q', 'v']。
    也可以是一个正则表达式，如'.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'。还可以使用通配符'all-linear'，
    匹配所有线性层层，除了输出层。

    -exclude_modules：用于指定要从 DaLoRA 中排除的模块名称或模块名称的正则表达式，即不使用 DaLora 的模块，与
    target_modules 作用相反。

    -bias：DaLoRA 的偏置参数训练选项，可以是 'none' 、 'all' 或 'lora_only' 。

    -layers_to_transform：用于指定要转换的层的索引，如果指定了该参数，PEFT将仅转换列表中指定的层索引，即仅对特定层进行
    模块替换，如果传递的是单个整数，PEFT将仅转换该索引处的层，该字段仅在target_modules是字符串列表时有效。

    -layers_pattern：用于指定层的模式名称，仅在layers_to_transform不为None且层模式不在常见层模式中时使用。
    该字段仅在target_modules是字符串列表时有效，layers_pattern应该针对模型的nn.ModuleList，通常被称为
    'layers'或'h'.

    -rank_pattern：用于将层名称或正则表达式映射到不同于默认r指定的rank值，例如，可以将特定层（如
    model.decoder.layers.0.encoder_attn.k_proj）的rank值设置为8。

    -alpha_pattern：用于将层名称或正则表达式映射到不同于默认lora_alpha指定的alpha值，例如，可以将特定层（如
    model.decoder.layers.0.encoder_attn.k_proj）的alpha值设置为32，即支持不同层使用不同的Lora alpha参数。
    '''
    r: int = field(
        default=8, metadata={"help": "Lora attention dimension"}
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None
    )
    Dalora_alpha: int = field(
        default=8, metadata={"help": "Lora alpha"}
    )
    Dalora_dropout: float = field(
        default=0.0, metadata={"help": "Lora dropout"}
    )
    bias: Literal["none", "all", "lora_only"] = field(
        default="none"
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict
    )

    def to_dict(self):
        """
        Returns the configuration for your adapter model as a dictionary. Removes runtime configurations.
        """
        rv = super().to_dict()
        rv.pop("runtime_config")
        return rv

    # 数据类创建后自动调用
    def __post_init__(self):
        self.peft_type = 'DaLora'
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")