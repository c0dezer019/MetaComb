#!/usr/bin/env python3
"""Comprehensive test for meta_comb_node.py with the provided PNG file."""

import sys
from pathlib import Path
from PIL import Image
import json
import logging
from meta_comb_node import MetaComb

logging.basicConfig(level=logging.INFO, format="%(message)s")

sys.path.insert(0, str(Path(__file__).parent))


def test_comprehensive():
    """Comprehensive test suite for metadata extraction."""
    png_path = Path(__file__).parent / ("example_image.png")

    img = Image.open(png_path)
    meta_comb = MetaComb()

    logging.info("=" * 80)
    logging.info("COMPREHENSIVE METADATA EXTRACTION TESTS")
    logging.info("=" * 80)

    # Test 1: Extract seed from PIL image
    logging.info("\n[TEST 1] Extract 'seed' from PIL image")
    result = meta_comb.comb_metadata(key="seed", image=img)
    logging.info(f"Result: {result}")
    assert result[0] == "1025463288989790", (
        f"Expected '1025463288989790', got '{result[0]}'"
    )
    logging.info("✓ PASSED")

    # Test 2: Extract other metadata fields
    logging.info("\n[TEST 2] Extract 'steps' from PIL image")
    result = meta_comb.comb_metadata(key="steps", image=img)
    logging.info(f"Result: {result}")
    assert result[0] == "20", f"Expected '20', got '{result[0]}'"
    logging.info("✓ PASSED")

    logging.info("\n[TEST 3] Extract 'cfg' from PIL image")
    result = meta_comb.comb_metadata(key="cfg", image=img)
    logging.info(f"Result: {result}")
    assert result[0] == "6.1", f"Expected '6.1', got '{result[0]}'"
    logging.info("✓ PASSED")

    logging.info("\n[TEST 4] Extract 'sampler_name' from PIL image")
    result = meta_comb.comb_metadata(key="sampler_name", image=img)
    logging.info(f"Result: {result}")
    assert result[0] == "dpmpp_2m", f"Expected 'dpmpp_2m', got '{result[0]}'"
    logging.info("✓ PASSED")

    logging.info("\n[TEST 5] Extract 'scheduler' from PIL image")
    result = meta_comb.comb_metadata(key="scheduler", image=img)
    logging.info(f"Result: {result}")
    assert result[0] == "karras", f"Expected 'karras', got '{result[0]}'"
    logging.info("✓ PASSED")

    # Test 3: Extract from raw metadata string (prompt)
    logging.info("\n[TEST 6] Extract 'seed' from raw metadata string (prompt)")
    result = meta_comb.comb_metadata(key="seed", metadata_raw=img.info["prompt"])
    logging.info(f"Result: {result}")
    assert result[0] == "1025463288989790", (
        f"Expected '1025463288989790', got '{result[0]}'"
    )
    logging.info("✓ PASSED")

    # Test 4: Search by node type
    logging.info("\n[TEST 7] Extract 'seed' from KSampler node type")
    result = meta_comb.comb_metadata(key="seed", image=img, node_type="KSampler")
    logging.info(f"Result: {result}")
    assert result[0] == "1025463288989790", (
        f"Expected '1025463288989790', got '{result[0]}'"
    )
    logging.info("✓ PASSED")

    # Test 5: Search by node title
    logging.info("\n[TEST 8] Extract 'seed' from 'KSampler' node title")
    result = meta_comb.comb_metadata(key="seed", image=img, node_title="KSampler")
    logging.info(f"Result: {result}")
    assert result[0] == "1025463288989790", (
        f"Expected '1025463288989790', got '{result[0]}'"
    )
    logging.info("✓ PASSED")

    # Test 8b: Return full node object when only node_title is provided
    logging.info("\n[TEST 8b] Return full node object for title 'KSampler' (no key)")
    result = meta_comb.comb_metadata(key="", image=img, node_title="KSampler")
    logging.info(f"Result: {result}")
    # Ensure result is valid JSON and contains the node's class_type
    parsed = json.loads(result[0])
    assert isinstance(parsed, dict), "Expected a JSON object for node title search"
    assert parsed.get("class_type") == "KSampler", (
        f"Expected class_type 'KSampler', got '{parsed.get('class_type')}'"
    )
    logging.info("✓ PASSED")

    # Test 9: Search by node type (no key -> return all nodes of that type)
    logging.info("\n[TEST 9] Return all nodes of type 'KSampler' (no key)")
    result = meta_comb.comb_metadata(key="", image=img, node_type="KSampler")
    logging.info(f"Result: {result}")
    parsed_list = json.loads(result[0])
    assert isinstance(parsed_list, list), "Expected a JSON array for node type search"
    assert any((obj.get("class_type") == "KSampler") for obj in parsed_list), (
        "Expected at least one KSampler node in results"
    )
    logging.info("✓ PASSED")

    # Test 6: Extract checkpoint name
    logging.info("\n[TEST 10] Extract 'ckpt_name' from image")
    result = meta_comb.comb_metadata(key="ckpt_name", image=img)
    logging.info(f"Result: {result}")
    assert "photon_v1.safetensors" in result[0], (
        f"Expected checkpoint name in result, got '{result[0]}'"
    )
    logging.info("✓ PASSED")

    # Test 7: Non-existent key
    logging.info("\n[TEST 10] Search for non-existent key")
    result = meta_comb.comb_metadata(key="nonexistent_key_12345", image=img)
    logging.info(f"Result: {result}")
    assert "not found" in result[0].lower(), (
        f"Expected 'not found' message, got '{result[0]}'"
    )
    logging.info("✓ PASSED")

    logging.info("\n" + "=" * 80)
    logging.info("ALL TESTS PASSED! ✓")
    logging.info("=" * 80)
    logging.info("\nThe node can successfully extract data from:")
    logging.info(f"  {png_path.name}")


if __name__ == "__main__":
    test_comprehensive()
