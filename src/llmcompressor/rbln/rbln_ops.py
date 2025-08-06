import os
from typing import Callable

import torch


def replace(original_func: Callable | list[Callable]):
    def decorator(custom_func: Callable):
        if os.getenv("USE_CUSTOM_OPS", "0") != "1":
            return custom_func

        if not isinstance(original_func, list):
            func_list = [original_func]
        else:
            func_list = original_func

        for func in func_list:
            module_path = func.__module__
            func_qualname = func.__qualname__
            if module_path == "torch":
                module = __import__(module_path, fromlist=[''])
                setattr(module, func.__name__, custom_func)
            elif module_path == "torch._C._linalg":
                module = __import__("torch.linalg", fromlist=[''])
                if "linalg_" in func_qualname:
                    func_qualname = func_qualname.split("linalg_")[-1]
                setattr(module, func_qualname, custom_func)
            else:
                raise NotImplementedError(f"Replacing module {module_path} not implemented yet.")
        return custom_func

    return decorator


@replace(torch.amin)
def torch_amin(x: torch.Tensor, dim: int | tuple[int] | None = None, keepdims: bool = False, *, out: torch.Tensor | None = None):
    """
    Custom implementation of torch.amin using torch.min.
    
    Args:
        x: Input tensor
        dim: Dimension(s) to reduce. Can be int, tuple of ints, or None (reduce all dims)
        keepdim: Whether to keep the reduced dimensions
        out: Optional output tensor to store the result
    
    Returns:
        Minimum values along the specified dimension(s)
    """
    if dim is None:
        result = torch.min(x)
        if out is not None:
            out.copy_(result)
            return out
        return result

    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = dim

    if not keepdims:
        dims = tuple(sorted(dims, reverse=True))
    
    result = x
    for d in dims:
        result = torch.min(result, dim=d, keepdim=keepdims).values
    
    if out is not None:
        out.copy_(result)
        return out
    
    return result


@replace(torch.amax)
def torch_amax(x: torch.Tensor, dim: int | tuple[int] | None = None, keepdims: bool = False, *, out: torch.Tensor | None = None):
    """
    Custom implementation of torch.amax using torch.max.
    
    Args:
        x: Input tensor
        dim: Dimension(s) to reduce. Can be int, tuple of ints, or None (reduce all dims)
        keepdim: Whether to keep the reduced dimensions
        out: Optional output tensor to store the result
    
    Returns:
        Maximum values along the specified dimension(s)
    """
    if dim is None:
        result = torch.max(x)
        if out is not None:
            out.copy_(result)
            return out
        return result

    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = dim

    if not keepdims:
        dims = tuple(sorted(dims, reverse=True))
    
    result = x
    for d in dims:
        result = torch.max(result, dim=d, keepdim=keepdims).values
    
    if out is not None:
        out.copy_(result)
        return out
    
    return result


@replace(torch.round)
def torch_round(x: torch.Tensor, *, decimals: int = 0, out: torch.Tensor | None = None):
    """
    Custom implementation of torch.round.
    
    Implements "round half to even" (banker's rounding) behavior.
    
    Args:
        x: Input tensor
        decimals: Number of decimal places to round to (default: 0)
        out: Optional output tensor to store the result
    
    Returns:
        Rounded tensor with same dtype as input
    """
    if decimals != 0:
        scale_factor = 10.0 ** decimals
        scaled_x = x * scale_factor
    else:
        scaled_x = x

    floor_x = torch.floor(scaled_x)
    frac_part = scaled_x - floor_x
    is_even = floor_x == 2 * torch.floor(floor_x / 2)
    is_half = torch.abs(frac_part - 0.5) < 1e-7
    should_round_up = frac_part >= 0.5
    should_round_up = torch.where(is_half, ~is_even, should_round_up)
    result = torch.where(should_round_up, floor_x + 1, floor_x)

    if decimals != 0:
        result = result / scale_factor
    if out is not None:
        out.copy_(result)
        return out
    
    return result


@replace(torch.linalg.cholesky)
def torch_cholesky(x: torch.Tensor, upper: bool = False):
    """
    Custom implementation of Cholesky decomposition using the Cholesky-Banachiewicz algorithm.
    
    Args:
        x: Symmetric positive definite matrix
        upper: If True, return upper triangular matrix U such that A = U.T @ U
               If False, return lower triangular matrix L such that A = L @ L.T
    
    Returns:
        Cholesky factor (lower or upper triangular matrix)
    """
    n = x.size(0)
    A = x.clone()
    if upper:
        for i in range(n):
            A[i, i] = torch.pow(A[i, i] - torch.sum(A[:i, i] ** 2), 0.5)
            for j in range(i + 1, n):
                A[i, j] = (A[i, j] - torch.sum(A[:i, i] * A[:i, j])) / A[i, i]
        U = torch.triu(A)
        return U
    else:
        for i in range(n):
            A[i, i] = torch.pow(A[i, i] - torch.sum(A[i, :i] ** 2), 0.5)
            for j in range(i + 1, n):
                A[j, i] = (A[j, i] - torch.sum(A[j, :i] * A[i, :i])) / A[i, i]
        L = torch.tril(A)
        return L


@replace(torch.cholesky_inverse)
def torch_cholesky_inverse(x: torch.Tensor, upper: bool = False):
    """
    Custom implementation of Cholesky inverse.
    
    Given a Cholesky factor L (or U), computes the inverse of the original matrix A
    where A = L @ L.T (or A = U.T @ U for upper triangular).
    
    Args:
        x: Cholesky factor (lower or upper triangular matrix)
        upper: If True, x is upper triangular (U), if False, x is lower triangular (L)
    
    Returns:
        Inverse of the original matrix A
    """
    n = x.size(0)
    if upper:
        U_inv = torch.zeros_like(x)
        for i in range(n - 1, -1, -1):
            U_inv[i, i] = torch.tensor(1.0, device=x.device, dtype=x.dtype) / x[i, i]
            for j in range(i + 1, n):
                sum_term = torch.sum(x[i, i + 1:j + 1] * U_inv[i + 1:j + 1, j])
                U_inv[i, j] = -sum_term / x[i, i]
        return U_inv @ U_inv.T
    else:
        L_inv = torch.zeros_like(x)
        for i in range(n):
            L_inv[i, i] = torch.tensor(1.0, device=x.device, dtype=x.dtype) / x[i, i]
            for j in range(i):
                sum_term = torch.sum(x[i, j:i] * L_inv[j:i, j])
                L_inv[i, j] = -sum_term / x[i, i]
        return L_inv.T @ L_inv
