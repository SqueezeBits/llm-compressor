from .rbln_utils import RBLNSubgraph
from .patch_compressed_tensors import apply_patch

# Apply the patch when the rbln module is imported
apply_patch()
