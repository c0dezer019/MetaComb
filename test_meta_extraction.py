#!/usr/bin/env python3
"""Test script for meta_comb_node.py metadata extraction."""

import json
import sys
from pathlib import Path
from PIL import Image

# Add the current directory to the path so we can import the node
sys.path.insert(0, str(Path(__file__).parent))

from meta_comb_node import MetaComb

def test_png_metadata_extraction():
    """Test extracting metadata from the PNG file."""

    png_path = Path(__file__).parent / "2025-12-02-desk-scene_Photon768x512-1025463288989790_00001_.png"

    print(f"Testing PNG file: {png_path}")
    print(f"File exists: {png_path.exists()}")
    print("-" * 80)

    # First, examine raw PNG metadata
    print("\n=== RAW PNG METADATA ===")
    img = Image.open(png_path)
    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")
    print(f"\nMetadata keys: {list(img.info.keys())}")

    for key, value in img.info.items():
        print(f"\n--- Key: {key} ---")
        if isinstance(value, str):
            if len(value) > 500:
                print(f"Type: str, Length: {len(value)}")
                print(f"First 500 chars: {value[:500]}")
                print(f"Last 100 chars: {value[-100:]}")

                # Try to parse as JSON
                try:
                    parsed = json.loads(value)
                    print(f"Parsed as JSON successfully!")
                    print(f"JSON type: {type(parsed)}")
                    if isinstance(parsed, dict):
                        print(f"Top-level keys: {list(parsed.keys())[:10]}")
                    elif isinstance(parsed, list):
                        print(f"List length: {len(parsed)}")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse as JSON: {e}")
            else:
                print(f"Value: {value}")
        else:
            print(f"Type: {type(value)}, Value: {value}")

    print("\n" + "=" * 80)
    print("\n=== TESTING MetaComb Node ===")

    # Test the MetaComb node with PIL Image
    meta_comb = MetaComb()

    # Test 1: Extract from PIL image with a simple key search
    print("\nTest 1: Extract from PIL image (searching for 'seed')")
    result = meta_comb.comb_metadata(key="seed", image=None, metadata_raw="")
    print(f"Result (no image): {result}")

    # Try to pass the PIL image directly (though this won't work as expected
    # since ComfyUI passes tensors, not PIL images)
    print("\nTest 2: Extract from PIL image directly")
    result = meta_comb.comb_metadata(key="seed", image=img)
    print(f"Result with PIL image: {result}")

    # Test 3: Extract raw metadata string
    print("\nTest 3: Extract from raw metadata string")
    if "workflow" in img.info:
        print("Found 'workflow' in metadata")
        result = meta_comb.comb_metadata(key="seed", metadata_raw=img.info["workflow"])
        print(f"Result: {result}")
    elif "prompt" in img.info:
        print("Found 'prompt' in metadata")
        result = meta_comb.comb_metadata(key="seed", metadata_raw=img.info["prompt"])
        print(f"Result: {result}")
    else:
        print("No 'workflow' or 'prompt' key found in metadata")
        # Try the first string value that looks like JSON
        for key, value in img.info.items():
            if isinstance(value, str) and value.strip().startswith("{"):
                print(f"Trying key: {key}")
                result = meta_comb.comb_metadata(key="seed", metadata_raw=value)
                print(f"Result: {result}")
                break

    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_png_metadata_extraction()
