from .rbln_utils import RBLNSubgraph
from .rbln_ops import *
from .patch_compressed_tensors import apply_patch

# Apply the patch when the rbln module is imported
apply_patch()
