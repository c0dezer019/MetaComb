#!/usr/bin/env python3
"""Test torch tensor input path for meta_comb_node.py."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from meta_comb_node import MetaComb  # noqa: E402

try:
    import torch  # type: ignore

    torch_available = True
except ImportError:
    torch_available = False
    torch = None  # type: ignore


def test_torch_tensor_input(capsys=None):  # type: ignore
    """Test that torch tensor input works correctly."""
    if not torch_available:
        return

    png_path = (
        Path(__file__).parent
        / "2025-12-02-desk-scene_Photon768x512-1025463288989790_00001_.png"
    )

    # Load image as PIL
    pil_img = Image.open(png_path)

    # Convert PIL image to ComfyUI-style torch tensor
    # ComfyUI format: [batch, height, width, channels], float32, range 0-1
    img_array = np.array(pil_img).astype(np.float32) / 255.0
    tensor_img = torch.from_numpy(img_array)[None, :]  # type: ignore

    meta_comb = MetaComb()

    # Test 1: Extract seed from torch tensor (include metadata_raw since tensor
    # conversion loses PNG metadata)
    result = meta_comb.comb_metadata(
        key="seed",
        image=tensor_img,
        metadata_raw=pil_img.info.get("prompt", ""),
    )
    expected = "1025463288989790"
    assert result[0] == expected, (
        f"Expected '{expected}', got '{result[0]}'"
    )

    # Test 2: Extract multiple fields
    result = meta_comb.comb_metadata(
        key="cfg",
        image=tensor_img,
        metadata_raw=pil_img.info.get("prompt", ""),
    )
    assert result[0] == "6.1", (
        f"Expected '6.1', got '{result[0]}'"
    )

    result = meta_comb.comb_metadata(
        key="sampler_name",
        image=tensor_img,
        metadata_raw=pil_img.info.get("prompt", ""),
    )
    expected_sampler = "dpmpp_2m"
    assert result[0] == expected_sampler, (
        f"Expected '{expected_sampler}', got '{result[0]}'"
    )


if __name__ == "__main__":
    test_torch_tensor_input()
