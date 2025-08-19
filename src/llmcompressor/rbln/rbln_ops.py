import os
from typing import Callable

import torch


def replace(*original_func: Callable):
    def decorator(custom_func: Callable):
        if os.getenv("USE_CUSTOM_OPS", "0") != "1":
            return custom_func
        for func in original_func:
            if hasattr(func, '__objclass__') and func.__objclass__ is torch._C.TensorBase:
                setattr(torch.Tensor, func.__name__, custom_func)
            elif hasattr(func, '__module__'):
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
            else:
                raise NotImplementedError(f"Cannot determine how to replace {func}. It doesn't appear to be a module function or tensor method.")
        return custom_func

    return decorator


# @replace(torch.sqrt, torch.Tensor.sqrt)
def torch_sqrt(x: torch.Tensor, *, out: torch.Tensor | None = None):
    """
    Custom implementation of torch.sqrt.
    """
    return torch.pow(x, 0.5)


@replace(torch.amin, torch.Tensor.amin)
def torch_amin(x: torch.Tensor, dim: int | tuple[int] | None = None, keepdim: bool = False, keepdims: bool = False, *, out: torch.Tensor | None = None):
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
    keepdim = keepdim or keepdims
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

    if not keepdim:
        dims = tuple(sorted(dims, reverse=True))
    
    result = x
    for d in dims:
        result = torch.min(result, dim=d, keepdim=keepdim).values
    
    if out is not None:
        out.copy_(result)
        return out
    
    return result


@replace(torch.amax, torch.Tensor.amax)
def torch_amax(x: torch.Tensor, dim: int | tuple[int] | None = None, keepdim: bool = False, keepdims: bool = False, *, out: torch.Tensor | None = None):
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
    keepdim = keepdim or keepdims
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

    if not keepdim:
        dims = tuple(sorted(dims, reverse=True))
    
    result = x
    for d in dims:
        result = torch.max(result, dim=d, keepdim=keepdim).values
    
    if out is not None:
        out.copy_(result)
        return out
    
    return result


@replace(torch.round, torch.Tensor.round)
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


def _gram_cols(B: torch.Tensor, row_chunk: int = 64) -> torch.Tensor:
    """
    Compute B^T B without @, using chunked outer-product accumulation to avoid
    materializing (n, d, d). Safe for large n.
      (B^T B) = sum_over_rows r_k^T r_k
    """
    if B.ndim != 2:
        raise ValueError("B must be 2D")
    n, d = B.shape
    G = B.new_zeros((d, d))
    for s in range(0, n, row_chunk):
        blk = B[s : s + row_chunk]              # (m, d)
        # (m,d,1) * (m,1,d) -> (m,d,d); sum over m -> (d,d)
        G = G + (blk.unsqueeze(2) * blk.unsqueeze(1)).sum(dim=0)
    return G


def _gram_rows(B: torch.Tensor, col_chunk: int = 64) -> torch.Tensor:
    """
    Compute B B^T without @, using chunked outer-product accumulation to avoid
    materializing (n, n, d). Safe for large d.
      (B B^T) = sum_over_cols c_k c_k^T
    """
    if B.ndim != 2:
        raise ValueError("B must be 2D")
    n, d = B.shape
    G = B.new_zeros((n, n))
    for s in range(0, d, col_chunk):
        blk = B[:, s : s + col_chunk]           # (n, m)
        # (n,1,m) * (1,n,m) -> (n,n,m); sum over m -> (n,n)
        G = G + (blk.unsqueeze(1) * blk.unsqueeze(0)).sum(dim=2)
    return G


def torch_cholesky_opt(x: torch.Tensor, upper: bool = False):
    """
    Cholesky (Banachiewicz, lower) without torch.linalg.
    If upper=True, returns the upper factor by transpose.
    """
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("x must be a square 2D tensor")

    A = x.clone().contiguous()
    n = A.shape[0]

    for i in range(n):
        if i > 0:
            # diag update: sqrt(a_ii - sum_{k<i} a_{ik}^2)
            s = torch.sum(A[i, :i] * A[i, :i], dim = 0) #torch.dot(A[i, :i], A[i, :i])
            A[i, i] = torch.pow(A[i, i] - s, 0.5) #torch.sqrt(A[i, i] - s)
        else:
            A[i, i] = torch.pow(A[i, i], 0.5) #torch.sqrt(A[i, i])

        if i + 1 < n:
            if i > 0:
                # below-diagonal column update (vectorized):
                # A[i+1:, i] = (A[i+1:, i] - A[i+1:, :i] @ A[i, :i]) / A[i, i]
                t = torch.sum(A[i + 1:, :i] * A[i, :i], dim = 1) #A[i + 1:, :i] @ A[i, :i]
                A[i + 1:, i] = (A[i + 1:, i] - t) / A[i, i]
            else:
                A[i + 1:, i] = A[i + 1:, i] / A[i, i]

    L = torch.tril(A)
    return L.T if upper else L


def torch_cholesky_inverse_opt(x: torch.Tensor, upper: bool = False):
    """
    Invert via triangular-inverse then sandwich, without torch.linalg / sqrt / dot / @.
    A^{-1} = L^{-T} L^{-1}  (lower),  A^{-1} = U^{-1} U^{-T} (upper)
    """
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("x must be a square 2D tensor")

    n = x.shape[0]

    if not upper:
        # x = L (lower)
        L = torch.tril(x).contiguous()
        Linv = torch.zeros_like(L)

        for i in range(n):
            if i > 0:
                # Linv[i, :i] = - (L[i, :i] @ Linv[:i, :i]) / L[i, i]
                # -> (i,) = sum_k L[i,k] * Linv[k,:]  (벡터화)
                t = (L[i, :i].unsqueeze(1) * Linv[:i, :i]).sum(dim=0)  # (i,)
                Linv[i, :i] = - t / L[i, i]
            Linv[i, i] = torch.pow(L[i, i], -1) #L[i, i].reciprocal()

        # A^{-1} = L^{-T} L^{-1} = (Linv^T) * Linv  (Gram of columns)
        #return (Linv.unsqueeze(2) * Linv.unsqueeze(1)).sum(dim=0)
        return _gram_cols(Linv, 64)
    else:
        # x = U (upper)
        U = torch.triu(x).contiguous()
        Uinv = torch.zeros_like(U)

        for i in range(n - 1, -1, -1):
            if i + 1 < n:
                # Uinv[i, i+1:] = - (U[i, i+1:] @ Uinv[i+1:, i+1:]) / U[i, i]
                r = U[i, i + 1:]                      # (m,)
                M = Uinv[i + 1:, i + 1:]              # (m,m) upper-tri
                t = (r.unsqueeze(1) * M).sum(dim=0)   # (m,)
                Uinv[i, i + 1:] = - t / U[i, i]
            Uinv[i, i] = torch.pow(U[i, i], -1) #U[i, i].reciprocal()

        # A^{-1} = U^{-1} U^{-T} = Uinv * (Uinv^T)  (Gram of rows)
        return _gram_rows(Uinv, 64)
