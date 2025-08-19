from typing import Optional, Tuple

import compressed_tensors.quantization.utils.helpers as ct_helpers
from compressed_tensors.quantization.utils.helpers import calculate_range, is_fp4
from compressed_tensors.quantization.quant_args import (
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationType,
)
import torch
from torch import FloatTensor, IntTensor, Tensor


def calculate_qparams_patched(
    min_vals: Tensor,
    max_vals: Tensor,
    quantization_args: QuantizationArgs,
    global_scale: Optional[Tensor] = None,
) -> Tuple[FloatTensor, IntTensor]:
    """
    :param min_vals: tensor of min value(s) to calculate scale(s) and zero point(s)
        from
    :param max_vals: tensor of max value(s) to calculate scale(s) and zero point(s)
        from
    :param quantization_args: settings to quantization
    :param global_scale: additional global scale to scale the locally generated scale
        currently only applied/supported for Fp4

    :return: tuple of the calculated scale(s) and zero point(s). For FP4, the calculated
        scale is of dtype FP8
    """
    # based on the implementations for consuming quantized values,
    # 0.0 must always be representable within the quantized range
    min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.max(max_vals, torch.zeros_like(max_vals))

    device = min_vals.device
    bit_min, bit_max = calculate_range(quantization_args, device)
    if bit_min.dtype != min_vals.dtype:
        bit_min = bit_min.to(min_vals.dtype)
    if bit_max.dtype != min_vals.dtype:
        bit_max = bit_max.to(min_vals.dtype)
    bit_range = bit_max - bit_min

    if is_fp4(quantization_args=quantization_args):
        zp_dtype = FP8_E4M3_DATA.dtype
    else:
        zp_dtype = quantization_args.pytorch_dtype()

    if quantization_args.symmetric:
        max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))

        if is_fp4(quantization_args=quantization_args) and global_scale is not None:
            # Conditionally scale the generated local scale by a global_scale
            scales = global_scale * (max_val_pos / FP4_E2M1_DATA.max)
            scales = torch.clamp(scales, max=FP8_E4M3_DATA.max, min=FP8_E4M3_DATA.min)
            scales = scales.to(FP8_E4M3_DATA.dtype)

        else:
            scales = max_val_pos / (float(bit_range) / 2)

        # TODO: in the case of MoEs, the global_scale may also be 0/need to be clamped
        if scales.dtype == FP8_E4M3_DATA.dtype:
            # torch.clamp not supported for FP8
            # use the next largest fp8 value from 0
            scales = torch.where(
                scales == 0,
                torch.tensor(0.125, dtype=FP8_E4M3_DATA.dtype, device=device),
                scales,
            )
        else:
            scales = torch.clamp(scales, min=torch.finfo(torch.float32).eps)

        zero_points = torch.zeros(scales.shape, device=device, dtype=min_vals.dtype)
    else:
        if is_fp4(quantization_args=quantization_args):
            raise NotImplementedError(
                "Asymmetric Quantization is not supported for FP4"
            )

        scales = (max_vals - min_vals) / float(bit_range)
        scales = torch.clamp(scales, min=torch.finfo(torch.float32).eps)
        zero_points = bit_min - (min_vals / scales)
        zero_points = torch.clamp(zero_points, bit_min, bit_max)

    # match zero-points to quantized type
    # if casting to int, use round instead of truncate
    if quantization_args.type == QuantizationType.INT:
        zero_points = torch.round(zero_points)
    zero_points = zero_points.to(zp_dtype)

    if scales.ndim == 0:
        scales = scales.reshape(1)
        zero_points = zero_points.reshape(1)

    return scales, zero_points

ct_helpers.calculate_qparams = calculate_qparams_patched
