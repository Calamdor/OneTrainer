from modules.util.mm_8bit import mm_8bit as mm_8bit
from modules.util.quantization_util import (
    quantize_fp8_axiswise,
    quantize_int8_axiswise,
)

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusers.quantizers.gguf.utils import GGUFLinear, dequantize_gguf_tensor

import gguf

UNQUANTIZED_TYPES = [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16]

# cuBLAS int8 GEMM requires M/N to be multiples of 8; fp8 GEMM on cuBLASLt
# requires multiples of 16. Both fire on the M dimension (the leading-row
# count of the LHS) for our LHS = activation, RHS = weight.T pattern.
# Pad to 16 so a single padding works for both paths.
_A8_ALIGN = 16


def _pad_a8_lhs(x_8: Tensor, x_scale: Tensor) -> tuple[Tensor, Tensor, int]:
    """Pad LHS along dim 0 up to a multiple of ``_A8_ALIGN`` for cuBLAS A8 GEMM.

    Returns (padded_x, padded_scale, original_n). Caller truncates the
    result back to ``original_n`` rows after the matmul.

    ``x_8`` is (N, K) quantized, ``x_scale`` is (N, 1) per-row scale.
    """
    n = x_8.shape[0]
    pad = (-n) % _A8_ALIGN
    if pad == 0:
        return x_8, x_scale, n
    x_8 = F.pad(x_8, (0, 0, 0, pad))
    x_scale = F.pad(x_scale, (0, 0, 0, pad))
    return x_8, x_scale, n


@torch.no_grad()
def int8_forward_axiswise(x: Tensor, weight: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    w_8, w_scale = quantize_int8_axiswise(weight, dim=-1)
    # cuBLAS int8 GEMM rejects shapes where N (LHS rows) isn't a multiple of
    # 8 with CUBLAS_STATUS_NOT_SUPPORTED. Fires on video models with frame
    # counts that produce non-aligned token totals (e.g. LTX-2.3 at 241
    # frames → 27×15×31 = 12555 tokens). Pad up, multiply, truncate.
    x_8, x_scale, n_orig = _pad_a8_lhs(x_8, x_scale)
    res = torch._int_mm(x_8, w_8.T)
    if res.shape[0] != n_orig:
        res = res[:n_orig]
        x_scale = x_scale[:n_orig]
    res_scaled = res.float().mul_(w_scale.T).mul_(x_scale).to(compute_dtype)
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

@torch.no_grad()
def fp8_forward_axiswise(x: Tensor, weight: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    x_8, x_scale = quantize_fp8_axiswise(x, dim=-1)
    w_8, w_scale = quantize_fp8_axiswise(weight, dim=-1)
    # cuBLASLt fp8 GEMM rejects non-aligned shapes with the misleading
    # "Only multiplication of row-major and column-major matrices is
    # supported" error. Same padding fix as int8 — pad LHS dim 0 to a
    # multiple of 16, matmul, truncate.
    x_8, x_scale, n_orig = _pad_a8_lhs(x_8, x_scale)
    one = torch.ones(1, device=x.device)
    res = torch._scaled_mm(x_8, w_8.T, scale_a=one, scale_b=one, out_dtype=torch.float)
    if res.shape[0] != n_orig:
        res = res[:n_orig]
        x_scale = x_scale[:n_orig]
    res_scaled = res.mul_(w_scale.T).mul_(x_scale).to(compute_dtype) #much faster than scaled by _scaled_mm
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

@torch.no_grad()
def int8_backward_axiswise(output: Tensor, weight: Tensor) -> Tensor:
    output_8, output_scale = quantize_int8_axiswise(output, dim=-1)
    w_8, w_scale = quantize_int8_axiswise(weight, dim=0)
    mm_res = mm_8bit(output_8.contiguous(), w_8)
    return mm_res.float().mul_(w_scale).mul_(output_scale).to(output.dtype)

@torch.no_grad()
def fp8_backward_axiswise(output: Tensor, weight: Tensor) -> Tensor:
    output_8, output_scale = quantize_fp8_axiswise(output, dim=-1)
    w_8, w_scale = quantize_fp8_axiswise(weight, dim=0)
    mm_res = mm_8bit(output_8.contiguous(), w_8)
    return mm_res.float().mul_(w_scale).mul_(output_scale).to(output.dtype)

class LinearGGUFIntA8RequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
        ctx.save_for_backward(weight)
        #axiswise performs better than tensorwise in tests, even though
        #it requires another requant during backward - but requant is cheap
        return int8_forward_axiswise(x, weight, bias, compute_dtype)

    @staticmethod
    def backward(ctx, output: Tensor):
        if ctx.needs_input_grad != (True, False, False, False):
            raise NotImplementedError("GGUF cannot be used for full finetuning")
        weight, = ctx.saved_tensors
        return int8_backward_axiswise(output, weight), None, None, None

class LinearGGUFFpA8RequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
        ctx.save_for_backward(weight)
        return fp8_forward_axiswise(x, weight, bias, compute_dtype)

    @staticmethod
    def backward(ctx, output: Tensor):
        if ctx.needs_input_grad != (True, False, False, False):
            raise NotImplementedError("GGUF cannot be used for full finetuning")
        weight, = ctx.saved_tensors
        return fp8_backward_axiswise(output, weight), None, None, None

class LinearGGUFA8(GGUFLinear):
    def __init__(self, dtype: torch.dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert dtype in [torch.int8, torch.float8_e4m3fn]
        self._dtype = dtype

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        assert not self.weight.requires_grad
        x = x_orig.reshape(-1, x_orig.shape[-1])
        w = dequantize_gguf_tensor(self.weight.detach())

        if x.shape[0] > 16 and hasattr(self.weight, 'quant_type') and self.weight.quant_type not in UNQUANTIZED_TYPES:
            if self._dtype == torch.int8:
                y = LinearGGUFIntA8RequantFunction.apply(x, w, self.bias, self.compute_dtype)
            else:
                y = LinearGGUFFpA8RequantFunction.apply(x, w, self.bias, self.compute_dtype)
        else:
            y = torch.nn.functional.linear(x, w, self.bias)

        return y.reshape(x_orig.shape[:-1] + (y.shape[-1], ))
