import warnings
from typing import Optional, Any

import GPUtil
import math
import torch
import torch.nn as nn
import numpy as np
import cupy as cp

from FastSVD.FastSVD import FastSVD

# Dalora base layer model
class DaLoraLayer:
    '''Dalora Layer base model, embedding Dalora is unsupported now'''
    r"""
        active_adapters (Union[List[`str`], `str`], *optional*):
            The name of the active adapter.
    """
    # indicates whether all adapters should be disabled
    _disable_adapters: bool = False

    # the currently active adapter(s)
    _active_adapter: str | list[str] = "default"

    # List all merged adapters
    merged_adapters: list[str] = []

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("Dalora_A", "Dalora_B")

    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "Dalora_alpha", "scaling", "Dalora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.Dalora_alpha = {}
        self.scaling = {}
        self.Dalora_dropout = nn.ModuleDict({})
        self.Dalora_A = nn.ModuleDict({})
        self.Dalora_B = nn.ModuleDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        # other parameters
        self.kwargs = kwargs

        # check base layer
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(
                "Only support nn.Linear layer now."
            )

        self.in_features = in_features
        self.out_features = out_features

        ''' FastSVD set '''
        self.method = 'compute'
        self.epoch = 20
        self.batch_size = 10000
        self.learning_rate = 0.01
        self.lamb = None
        self.regularization = None
        self.cross_regu = None

    @property
    def bias(self) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return base_layer.bias

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str | list[str]:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    def _get_available_adapters(self) -> set[str]:
        """Return all adapter names that can be found on this module."""
        adapters = set()
        for layer_name in self.adapter_layer_names:
            module = getattr(self, layer_name)
            if not isinstance(module, (nn.ModuleDict, nn.ParameterDict)):
                continue
            adapters.update(set(module.keys()))
        return adapters

    @property
    def active_adapters(self):
        # to list of str
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        return self.active_adapter

    # whether enable all adapters or not
    def enable_adapters(self, enabled: bool) -> None:
        """
        Toggle the enabling and disabling of adapters
        Takes care of setting the requires_grad flag for the adapter weights.
        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            # disable grads on all adapter layers
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.requires_grad_(False)
            self._disable_adapters = True

    def _all_available_adapter_names(self) -> list[str]:
        """Return a sorted list of all available adapter names"""
        adapter_names = set()
        for name in self.adapter_layer_names + self.other_param_names:
            # we check each possible attribute and if it's a dict or ModuleDict, we assume that the keys are the adapter
            # names
            attr = getattr(self, name)
            if hasattr(attr, "keys"):
                adapter_names.update(attr.keys())
        return sorted(adapter_names)

    # get Dalora layer's base layer
    def get_base_layer(self) -> nn.Module:
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer

    # while be called in subclass's `__init__` function
    def update_layer(self, adapter_name, r, inputs, Dalora_alpha, Dalora_dropout):
        self.r[adapter_name] = r
        self.Dalora_alpha[adapter_name] = Dalora_alpha
        self.Dalora_dropout.update(nn.ModuleDict({adapter_name: nn.Dropout(p=Dalora_dropout)}))

        # Actual trainable parameters
        self.Dalora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.Dalora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

        # scaling
        self.scaling[adapter_name] = Dalora_alpha / r

        # Dalora matrix init
        self.matrix_init(adapter_name, inputs)
        self._move_adapter_to_device_of_base_layer(adapter_name)

        # subclass method
        self.set_adapter(self.active_adapters)

    # Dalora matrix init
    def matrix_init(self, adapter_name, inputs):
        '''
        :param adapter_name: current adapter name
        :param inputs: sampling data as target's inputs, shape of [B, L, dim]
        '''

        ''' prepare W matirx '''
        weight = self.get_base_layer().weight  # [dim_out, dim_in]
        dtype = weight.dtype

        # data type cast is needed while using FastSVD class
        if dtype != torch.float64:
            cast_weight = weight.to(torch.float64)
        else:
            cast_weight = weight

        '''
        这个位置出现了一些问题
        '''
        # print(self.base_layer)
        W = cast_weight.cpu().numpy().T  # [dim_in, dim_out]
        del cast_weight

        ''' prepare X matirx '''
        X = inputs.view(-1, inputs.shape[-1])  # [B, L, dim_in] -> [B * L, dim_in]
        del inputs  # 节省内存

        if X.dtype != torch.float64:
            X = X.to(torch.float64)

        # 查看GPU占用
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(
                f"{gpu.memoryUsed}MB / {gpu.memoryTotal}MB"
            )
            break

        X = X.cpu().numpy()

        ''' FastSVD '''
        rank = self.r[adapter_name]
        fast = FastSVD(X=X, W=W, r=rank, batch_size=self.batch_size, lr=self.learning_rate,
                       regu=self.regularization, lamb=self.lamb, cross=self.cross_regu)

        # train or compute matrix directly
        if self.method == 'train':
            fast.train(self.epoch)
            U = fast.U.data
            S = fast.Sr
            VT = fast.VTr
        else:
            U, S, VT = fast.ComputeU()
            S = S[:rank, :rank]  # [r, r]
            U = U[:, :rank]  # [BL, r]
            VT = VT[:rank, :]  # [r, dim_out]

        "<<< 下面对X.T @ X逆矩阵的计算进行修正 >>>"
        '''
        这里需要对于X.T @ X进行分析，如果X.T @ X不可逆，则数值将会出现不稳定性，这时可以使用修正方式：
        1.对于X.T @ X加上对角线元素平均值的缩放因子，不断进行分解，直到能够分解。
        2.对X.T @ X先进行特征值分解，将对角线上的0值替换为所有特征值平均值的缩放因子。
        '''
        np.set_printoptions(precision=4)

        # 方法1
        XTX = X.T @ X

        # 这里GPU加速
        # xU, xS, xVT = np.linalg.svd(XTX)
        # 指定设备并执行计算
        with cp.cuda.Device(cp.cuda.runtime.getDeviceCount() - 1):
            XTX = cp.asarray(XTX)
            xU, xS, xVT = cp.linalg.svd(XTX)
            xU = cp.asnumpy(xU)
            xS = cp.asnumpy(xS)
            xVT = cp.asnumpy(xVT)

        nzeros = np.sum(xS > 0)
        print("rank of X.T * X:", nzeros)
        print("values:", xS)
        print('\n')

        # 打开文件并追加内容
        with open("/root/autodl-tmp/code/result.txt", "a", encoding="utf-8") as file:
            file.write(f"rank of X.T * X:{nzeros}\n")
            file.write(f"values:{xS}\n")

        # # 修正
        # if nzeros < X.shape[1]:
        #     esum = np.sum(xS) / X.shape[1] * 0.001
        #     print("Eta:", esum)
        #     xS[xS < esum] = esum

        esum = np.sum(xS) / X.shape[1] / math.sqrt(X.shape[0] * X.shape[1])
        print("Eta:", esum)
        xS[xS < esum] = esum

        xS = 1 / xS
        invM = xVT.T @ np.diag(xS) @ xU.T  # 修正后的逆矩阵

        # # 方法2
        # evalues, evectors = np.linalg.eig(XTX)
        # evectors = np.abs(evectors)
        # evalues = np.abs(evalues)

        # nzeros = np.sum(evalues > 0)
        # print("rank of X.T * X:", nzeros)
        # print("values:", evalues)
        # print("vectors:", evectors)

        # esum = np.sum(evalues) / X.shape[1] * 0.01
        # print("Eta:", esum)

        # evalues[evalues < esum] = esum
        # evalues = 1 / evalues
        # invM = evectors @ np.diag(evalues) @ evectors.T  # 修正后的逆矩阵

        nan_count = np.sum(np.isnan(invM))
        if nan_count > 0:
            raise (f'Nan value. Number:{nan_count}')

        # print("X.T * X", XTX, XTX.dtype)
        # print("inv(X.T * X):", invM, invM.dtype)
        # print("original * inv:", XTX @ invM)
        # print('\n')

        nan_count = np.sum(np.isnan(invM))
        if nan_count > 0:
            raise (f'Nan value. Number:{nan_count}')

        '''
        验证分解效果
        '''
        XW = X @ W

        # 查看相对逼近程度
        delta = np.abs((XW - U @ S @ VT) / (XW + 0.000001))
        m_max = np.max(delta)
        m_middle = np.median(delta)
        m_mean = np.sum(delta) / (delta.shape[0] * delta.shape[1])
        print("SVD(XW) Max bias with rank limitation:", m_max)
        print("SVD(XW) Middle bias with rank limitation:", m_middle)
        print("SVD(XW) Average bias with rank limitation:", m_mean)
        print('\n')

        # 打开文件并追加内容
        with open("/root/autodl-tmp/code/result.txt", "a", encoding="utf-8") as file:
            file.write(f"SVD(XW) Max bias with rank limitation:{m_max}\n")
            file.write(f"SVD(XW) Middlebias with rank limitation:{m_middle}\n")
            file.write(f"SVD(XW) Average bias with rank limitation::{m_mean}\n\n")


        "<<< 上面对X.T @ X逆矩阵的计算进行修正 >>>"

        # compute B & A
        S /= self.scaling[adapter_name]
        A = invM @ X.T @ U @ np.sqrt(S)  # [dim_in, dim_in] [dim_in, BL] [BL, r] [r, r] -> [dim_in, r]
        B = np.sqrt(S) @ VT  # [r, r] [r, dim_out] -> [r, dim_out]

        "<<< 下面验证|X * hatW - X * W|的分解效果 >>>"
        XWh = X @ A @ B * self.scaling[adapter_name]  # 注意这里将缩放因子还原
        # print("X * hatW:", XWh)
        # print("X * W:", XW)
        # print('\n')

        delta = np.abs(XWh - XW)
        # print("Delta:", delta)
        # print("|X * hatW - X * W|:")
        # print("max bias:", np.max(delta))
        # print("middle bias:", np.median(delta))
        # print("mean bias:", np.sum(delta) / (delta.shape[0] * delta.shape[1]))
        # print('\n')

        cdelta = np.abs(delta / (XW + 0.000001))
        # print("cDelta:", cdelta)
        print("|X * hatW - X * W|:")
        # print("max compared bias:", np.max(cdelta))
        print("middle compared bias:", np.median(cdelta))
        # print("mean compared bias:", np.sum(cdelta) / (delta.shape[0] * delta.shape[1]))
        print('\n')

        # 打开文件并追加内容
        with open("/root/autodl-tmp/code/result.txt", "a", encoding="utf-8") as file:
            file.write(f"|X * hatW - X * W| middle compared bias:{np.median(cdelta)}\n\n")

        "<<< 上面验证|X * hatW - X * W|的分解效果 >>>"

        # to tensor
        original_weight = weight.data.float()

        # w - delta W，高精度计算
        Dalora_A = torch.tensor(A.T, dtype=original_weight.dtype).to(weight.device)  # [r, dim_in]
        Dalora_B = torch.tensor(B.T, dtype=original_weight.dtype).to(weight.device)  # [dim_out, r]
        residual_weight = original_weight - (self.scaling[adapter_name] * Dalora_B) @ Dalora_A 

        self.Dalora_A[adapter_name].weight.data = Dalora_A 
        self.Dalora_B[adapter_name].weight.data = Dalora_B 
        self.get_base_layer().weight.data = residual_weight.to(dtype)

        # print("A max & min:", torch.max(Dalora_A), torch.min(Dalora_A))
        # print("B max & min:", torch.max(Dalora_B), torch.min(Dalora_B))
        
        now_weight = weight.data + (self.scaling[adapter_name] * Dalora_B) @ Dalora_A
        print("Weight compare:", torch.abs(original_weight - now_weight) / original_weight)

    # move adapter to device which base layer is on
    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        """
        Move the adapter of the given name to the device of the base layer.
        """
        if device is None:
            # check weight and qweight (for GPTQ)
            for weight_name in ("weight", "qweight"):
                weight = getattr(self.get_base_layer(), weight_name, None)
                if weight is not None:
                    device = weight.device
                    dtype = weight.dtype
                    break
            else:
                # no break encountered: could not determine the device
                return

        meta = torch.device("meta")
        for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict)):
                continue
            if adapter_name not in adapter_layer:
                continue
            if any(p.device == meta for p in adapter_layer.parameters()):
                continue

            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)
            else:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)

    # set current adapter as trainable while set other adapters as un-trainable
    def set_adapter(self, adapter_names: str | list[str]) -> None:
        """
        Set the active adapter(s).
        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True).
        If this is not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```
        Args:
            adapter_name (`str` or `List[str]`): Name of the adapter(s) to be activated.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                '''
                Note: It is possible that not a single layer is called with requires_grad_(True) here. 
                This may happen if a completely different adapter layer is being activated.
                '''
                if key in adapter_names:
                    layer.requires_grad_(True)
                else:
                    layer.requires_grad_(False)

        self._active_adapter = adapter_names

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Delete an adapter from the layer
        This should be called on all adapter layers, or else we will get an inconsistent state.
        This method will also set a new active adapter if the deleted adapter was an active adapter. It is important
        that the new adapter is chosen in a deterministic way, so that the same adapter is chosen on all layers.
        Args:
            adapter_name (`str`): The name of the adapter to delete
        """
        for attr in self.adapter_layer_names + self.other_param_names:
            if adapter_name in getattr(self, attr):
                del getattr(self, attr)[adapter_name]

        if adapter_name in self.active_adapters:
            # choose a new active adapter
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                # no active adapters left, set a new default adapter
                # here we get the list of all adapters existing adapter names and choose the first one
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to "
                        f"{new_active_adapter}."
                    )
                    self.set_adapter(remaining_adapters[0])

    # All adapters mixed for batch inference
    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.Dalora_A.keys():
                continue

            Dalora_A = self.Dalora_A[active_adapter]
            Dalora_B = self.Dalora_B[active_adapter]
            dropout = self.Dalora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            '''
            getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            layer output
            '''
            sub_batch = x[sub_batch_indices_list[i]].to(Dalora_A.weight.dtype)
            lora_output = Dalora_B(Dalora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result

# Dalora Linear layer
class Linear(nn.Module, DaLoraLayer):
    # DaLora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        inputs,
        adapter_name: str,
        r: int = 8,
        Dalora_alpha: int = 32,
        Dalora_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        '''
        :param base_layer: original layer to be tuned
        :param inputs: sampling data as layer's input
        :param adapter_name: current adapter name
        :param r: Dalora rank
        :param Dalora_alpha: Dalora alpha
        :param Dalora_dropout: lora dropout probability
        :param kwargs: other parameters
        '''
        super().__init__()
        DaLoraLayer.__init__(self, base_layer, **kwargs)

        self._active_adapter = adapter_name

        # construct or update a Dalora linear layer, father class method
        self.update_layer(
            adapter_name,
            r,
            inputs=inputs,
            Dalora_alpha=Dalora_alpha,
            Dalora_dropout=Dalora_dropout,
        )

    # compute ans return delta W
    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.
        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.Dalora_B[adapter].weight.device
        dtype = self.Dalora_B[adapter].weight.dtype

        '''
        In case users wants to merge the adapter weights that are in (b)float16 while being on CPU, 
        we need to cast the weights to float32, perform the merge and then cast back to (b)float16 
        because some CPUs have slow bf16/fp16 matmuls.
        '''
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.Dalora_A[adapter].weight
        weight_B = self.Dalora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = (weight_B @ weight_A) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.Dalora_A[adapter].weight.data = weight_A.to(dtype)
            self.Dalora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    # merge all adapter's weight (delta W) into base layer's weight (W)
    def merge(self, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights
        Args:
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)

        # no adapter to merge
        if not adapter_names:
            return

        # merging
        for active_adapter in adapter_names:
            if active_adapter in self.Dalora_A.keys():
                base_layer = self.get_base_layer()
                delta_weight = self.get_delta_weight(active_adapter)
                base_layer.weight.data += delta_weight
                self.merged_adapters.append(active_adapter)

    # detach all adapter's weight (delta W) from merged base layer's weight (W + delta W)
    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        # unmerging
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.Dalora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight

    # different ways to forward
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.Dalora_A.keys():
                    continue
                Dalora_A = self.Dalora_A[active_adapter]
                Dalora_B = self.Dalora_B[active_adapter]
                dropout = self.Dalora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                x = x.to(Dalora_A.weight.dtype)
                result = result + Dalora_B(Dalora_A(dropout(x))) * scaling
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "Dalora." + rep

# 返回当前还没有和base layer进行merge的adapter名称
def check_adapters_to_merge(module: DaLoraLayer, adapter_names: Optional[list[str]] = None) -> list[str]:
    """
    Helper function to check which adapters should be merged.
    Only return those adapters that are not already merged. Give a warning if some or all of the adapters are already
    merged.
    """
    if adapter_names is None:
        adapter_names = module.active_adapters
    if isinstance(adapter_names, str):
        raise ValueError(f"adapter_names should be a list of strings, got {adapter_names!r}.")

    if module.merged:
        merged_adapters = set(module.merged_adapters)
        adapter_names = [name for name in adapter_names if name not in merged_adapters]

        if adapter_names:
            warnings.warn(
                f"Already following adapters were merged {','.join(module.merged_adapters)}. "
                f"You are now additionally merging {','.join(adapter_names)}."
            )
        else:
            warnings.warn("All adapters are already merged, nothing to do.")

    return adapter_names