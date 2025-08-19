"""
Patch for compressed_tensors package to fix dtype conversion issues.
This patch is automatically applied when the rbln module is imported.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def find_compressed_tensors_path() -> Optional[Path]:
    """Find the path to the compressed_tensors package."""
    try:
        import compressed_tensors
        return Path(compressed_tensors.__file__).parent
    except ImportError:
        return None


def apply_patch():
    """Apply the patch to compressed_tensors helpers.py file."""
    compressed_tensors_path = find_compressed_tensors_path()
    if not compressed_tensors_path:
        print("Warning: compressed_tensors package not found, skipping patch")
        return
    
    helpers_path = compressed_tensors_path / "quantization" / "utils" / "helpers.py"
    if not helpers_path.exists():
        print(f"Warning: helpers.py not found at {helpers_path}, skipping patch")
        return
    
    # Read the original file
    with open(helpers_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if patch is already applied
    if "# PATCHED: Enhanced dtype conversion" in content:
        print("Patch already applied to compressed_tensors")
        return
    
    # Find the target lines to patch
    target_lines = [
        "    bit_min, bit_max = calculate_range(quantization_args, device)",
        "    bit_range = bit_max - bit_min"
    ]
    
    # Check if all target lines exist
    for line in target_lines:
        if line not in content:
            print(f"Warning: Target line not found: {line}")
            return
    
    # Create the patched version
    patched_section = """    bit_min, bit_max = calculate_range(quantization_args, device)
    # PATCHED: Enhanced dtype conversion to ensure compatibility
    if bit_min.dtype != min_vals.dtype:
        bit_min = bit_min.to(min_vals.dtype)
    if bit_max.dtype != min_vals.dtype:
        bit_max = bit_max.to(min_vals.dtype)
    bit_range = bit_max - bit_min"""
    
    # Replace the original section
    original_section = "\n".join(target_lines)
    content = content.replace(original_section, patched_section)
    
    # Write the patched file
    with open(helpers_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Successfully applied patch to {helpers_path}")


# Apply the patch when this module is imported
apply_patch()
