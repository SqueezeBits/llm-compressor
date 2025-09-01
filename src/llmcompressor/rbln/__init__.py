import functools
import torch

@functools.cache
def is_rbln_available():
    try:
        import rebel
    except Exception as e:
        return False
    return hasattr(torch, "rbln")

if is_rbln_available():
    from .rbln_utils import RBLNSubgraph, ENFORCE_EAGER
    from .rbln_ops import *
    from .patch_compressed_tensors import *
else:
    ENFORCE_EAGER = True
