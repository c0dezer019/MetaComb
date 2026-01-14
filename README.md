# MetaComb

A ComfyUI custom node for extracting and searching metadata from ComfyUI workflow images.

## Overview

MetaComb is a utility node that allows you to extract metadata and parameter values from ComfyUI-generated PNG images. It can parse workflow data embedded in images and search for specific parameters by key name, node type, or node title.

## Features

- **Multiple Input Types**: Supports PIL Images, PyTorch tensors, and raw metadata strings
- **Flexible Search**: Find parameters by key name, node type, or node title
- **Workflow Parsing**: Extracts both `workflow` and `prompt` data from ComfyUI PNG metadata
- **Deep Search**: Recursively searches nested data structures to find parameter values
- **ComfyUI Integration**: Fully integrated as a native ComfyUI node

## Installation

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes/
```

2. Clone or copy this repository:
```bash
git clone <repository-url> meta_comb
# or copy the meta_comb folder directly
```

3. Restart ComfyUI

The node will appear in the "utils" category as "Meta Comb".

## Usage

### Basic Usage

The MetaComb node requires at least one of the following inputs: `key`, `node_title`, or `node_type`.

- **key** (STRING, optional): The metadata key you want to extract (e.g., "seed", "steps", "cfg", "sampler_name"). Provide this to extract a specific parameter value.
- **node_title** (STRING, optional): If provided and `key` is empty, the node returns the entire node object matching this title. If `key` is provided along with `node_title`, it searches only within nodes that match the title.
- **node_type** (STRING, optional): If provided and `key` is empty, the node returns all node objects of this type. If `key` is provided along with `node_type`, it searches only within nodes of that type.

Other optional parameters:
- **image** (IMAGE): A ComfyUI image tensor or PIL Image containing workflow metadata
- **metadata_raw** (STRING): Raw JSON metadata string as an alternative to image input
- **search_workflow** (BOOLEAN): Search in workflow data instead of prompt data (default: False)

### Examples

#### Example 1: Extract Seed from Image
```python
# In ComfyUI workflow
key: "seed"
image: [connect your image output]
# Returns: "1025463288989790"
```

#### Example 2: Extract CFG Scale
```python
key: "cfg"
image: [connect your image output]
# Returns: "6.1"
```

#### Example 3: Search by Node Type
```python
key: "seed"
image: [connect your image output]
node_type: "KSampler"
# Returns: "1025463288989790"
```

#### Example 4: Search by Node Title
```python
key: "ckpt_name"
image: [connect your image output]
node_title: "Load Checkpoint"
# Returns: "photon_v1.safetensors"
```

#### Example 5: Use Raw Metadata String
```python
key: "steps"
metadata_raw: '{"3": {"inputs": {"steps": 20}}}'
# Returns: "20"
```

### Return Values

- **Single Match**: Returns the value as a string
- **Multiple Matches**: Returns a JSON array of all matching values
- **No Match**: Returns "Key '[key]' not found"
- **No Data**: Returns "No workflow data found"

## Common Metadata Keys

Here are some common keys you can extract from ComfyUI images:

| Key | Description | Example Value |
|-----|-------------|---------------|
| `seed` | Random seed used for generation | `1025463288989790` |
| `steps` | Number of sampling steps | `20` |
| `cfg` | Classifier-free guidance scale | `6.1` |
| `sampler_name` | Sampler algorithm used | `dpmpp_2m` |
| `scheduler` | Scheduler type | `karras` |
| `ckpt_name` | Checkpoint/model name | `photon_v1.safetensors` |
| `denoise` | Denoising strength | `1.0` |
| `width` | Image width | `768` |
| `height` | Image height | `512` |

## Technical Details

### Input Formats

**PyTorch Tensor Format**: ComfyUI-style tensors with shape `[batch, height, width, channels]`, float32, range 0-1

**PIL Image**: Standard PIL Image objects with PNG metadata

**Raw Metadata**: JSON strings containing workflow or prompt data

### Metadata Extraction Process

1. If `metadata_raw` is provided, parse it as JSON
2. If an `image` is provided, extract PNG metadata:
   - Convert torch tensors to PIL Images if needed
   - Extract `workflow` and `prompt` keys from PNG info
3. Search the appropriate data structure (workflow or prompt)
4. Apply filters if `node_title` or `node_type` are specified
5. Return matching values

### Search Behavior

- **No Filters**: Returns the first match found across all nodes
- **With node_type**: Returns all matches from nodes of that type
- **With node_title**: Returns matches from nodes with that title
- **With Both**: Returns matches from nodes matching both criteria

## Development

### Project Structure

```
meta_comb/
├── __init__.py                 # Node registration
├── meta_comb_node.py          # Main node implementation
├── test_comprehensive.py      # Comprehensive test suite
├── test_meta_extraction.py    # Metadata extraction tests
├── test_torch_tensor.py       # PyTorch tensor input tests
├── test_final_verification.py # Final verification tests
└── 2025-12-02-desk-scene_*.png # Test image with workflow data
```

### Running Tests

Run the test suite to verify functionality:

```bash
# Run comprehensive tests
python test_comprehensive.py

# Run tensor input tests
python test_torch_tensor.py

# Run metadata extraction tests
python test_meta_extraction.py

# Run final verification tests
python test_final_verification.py
```

All tests should pass with output indicating successful metadata extraction.

### Dependencies

- **Required**:
  - Python 3.7+
  - PIL (Pillow)
  - ComfyUI

- **Optional**:
  - PyTorch (for tensor input support)
  - NumPy (for tensor conversion)

## Use Cases

1. **Workflow Analysis**: Extract parameters from generated images to understand what settings were used
2. **Batch Processing**: Automatically read generation parameters from a collection of images
3. **Quality Control**: Verify that images were generated with expected parameters
4. **Debugging**: Inspect workflow data to troubleshoot generation issues
5. **Automation**: Build workflows that adapt based on parameters from previous generations
6. **Documentation**: Automatically document generation settings from output images

## Troubleshooting

### "No workflow data found"
- Ensure the image was generated by ComfyUI and contains embedded workflow metadata
- Some image formats or processing may strip metadata
- Try using the `metadata_raw` input with extracted JSON data

### "Key '[key]' not found"
- The specified key doesn't exist in the workflow data
- Try without filters first to see if the key exists elsewhere
- Check the exact key name (case-sensitive)

### Empty or Unexpected Results
- Use `search_workflow: true` to search workflow data instead of prompt data
- Try specifying `node_type` or `node_title` to narrow the search
- Some parameters may be nested deeper in the structure

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Changelog

### Version 1.0.0
- Initial release
- Support for PIL Image and PyTorch tensor inputs
- Flexible search by key, node type, and node title
- Comprehensive test coverage
- Deep recursive search capability
