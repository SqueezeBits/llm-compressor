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
    from .rbln_envs import ENFORCE_EAGER, USE_CUSTOM_OPS
    from .rbln_subgraph import RBLNSubgraph
    from .rbln_ops import *
    from .patch_compressed_tensors import *
else:
    ENFORCE_EAGER = True
    USE_CUSTOM_OPS = False
