import json
from typing import Any, Dict, List, Optional, cast
from PIL import Image


class MetaComb:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {},
            "optional": {
                "key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "filepath": ("STRING", {"forceInput": True}),
                "metadata_raw": (
                    "STRING",
                    {"default": "", "multiline": True}
                ),
                "node_title": ("STRING", {"default": ""}),
                "node_type": ("STRING", {"default": ""}),
                "search_workflow": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "comb_metadata"
    CATEGORY = "utils"

    def comb_metadata(
        self,
        key: str,
        image: Optional[Any] = None,
        filepath: str = "",
        metadata_raw: str = "",
        node_title: str = "",
        node_type: str = "",
        search_workflow: bool = False,
        prompt: Optional[Dict[str, Any]] = None,
        extra_pnginfo: Optional[Dict[str, Any]] = None,
    ) -> tuple[str]:
        """Extract metadata from ComfyUI PNG workflow data or raw metadata
        string."""

        workflow_data = None

        # Try to get workflow data from multiple sources
        # Priority 1: Extract from PNG file path (if provided)
        if filepath:
            workflow_data = self._extract_from_file_path(filepath)
            if not workflow_data:
                return (
                    "Error: Filepath provided but no metadata found in file",
                )

        # Priority 2: Extract from image tensor
        # (note: usually stripped by Load Image node)
        if not workflow_data and image is not None:
            workflow_data = self._extract_from_image(image)
            if not workflow_data:
                # Don't fail immediately; allow metadata_raw or prompt to be used as fallback
                print("MetaComb: Image provided but no metadata found in image; continuing to other sources")
                workflow_data = None

        # Priority 3: Raw metadata string
        if not workflow_data and metadata_raw:
            workflow_data = self._parse_raw_metadata(metadata_raw)
            if not workflow_data:
                return (
                    "Error: Raw metadata provided but could not be parsed",
                )

        # Priority 4: Current execution prompt (fallback to current workflow)
        if not workflow_data and prompt is not None:
            workflow_data = {"prompt": prompt}
            if extra_pnginfo and "workflow" in extra_pnginfo:
                workflow_data["workflow"] = extra_pnginfo["workflow"]

        if not workflow_data:
            return ("No workflow data found",)

        # Choose search scope
        search_scope = workflow_data.get(
            "workflow" if search_workflow else "prompt", {}
        )

        # If scope is empty, try the other one
        if not search_scope:
            search_scope = workflow_data.get(
                "prompt" if search_workflow else "workflow", {}
            )

        # If still empty, search the entire data structure
        if not search_scope:
            search_scope = workflow_data

        # Require at least one of key, node_title, or node_type
        if not (key and str(key).strip()) and not node_title and not node_type:
            return (
                "Error: Please provide at least one of key, node_title, or node_type",
            )

        # Perform search based on parameters
        results = self._search_nodes(search_scope, key, node_title, node_type)

        # Format output
        if not results:
            # Tailored messages when searching for nodes instead of a key
            if not (key and str(key).strip()):
                if node_title and node_type:
                    return (f"No nodes found matching title '{node_title}' and type '{node_type}'",)
                if node_title:
                    return (f"No node found with title '{node_title}'",)
                if node_type:
                    return (f"No nodes found of type '{node_type}'",)
            return (f"Key '{key}' not found",)

        # If key was not provided, handle node object returns
        if not (key and str(key).strip()):
            # node_type only -> always return an array (even if single)
            if node_type and not node_title:
                return (json.dumps(results, indent=2),)

            # node_title only -> if single, return object, else array
            if node_title and not node_type:
                if len(results) == 1:
                    return (json.dumps(results[0], indent=2),)
                return (json.dumps(results, indent=2),)

            # both title and type -> return array
            return (json.dumps(results, indent=2),)

        # If result(s) are values, return string(s)
        if len(results) == 1:
            return (str(results[0]),)

        # Multiple results
        return (json.dumps(results, indent=2),)

    def _parse_raw_metadata(self, metadata_raw: str) -> Dict[str, Any]:
        """Parse raw metadata string into dict."""
        if not metadata_raw:
            return {}

        metadata_raw = metadata_raw.strip()

        if not metadata_raw:
            return {}

        # Try direct JSON parsing first
        try:
            parsed = json.loads(metadata_raw)
            # If it's a dict, return it directly
            if isinstance(parsed, dict):
                return cast(Dict[str, Any], parsed)
            # If it's a list or other type, wrap it in a dict
            return {"data": parsed}
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON
            pass

        # Try to extract JSON from text (find first { and last })
        try:
            start = metadata_raw.find("{")
            end = metadata_raw.rfind("}")
            if start != -1 and end != -1:
                json_str = metadata_raw[start:end + 1]
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return cast(Dict[str, Any], parsed)
                return {"data": parsed}
        except json.JSONDecodeError:
            pass

        # Try to handle double-encoded JSON (JSON string containing JSON)
        try:
            # If the string looks like it might be double-encoded
            if metadata_raw.startswith('"') and metadata_raw.endswith('"'):
                decoded = json.loads(metadata_raw)
                if isinstance(decoded, str):
                    return self._parse_raw_metadata(decoded)
                elif isinstance(decoded, dict):
                    return cast(Dict[str, Any], decoded)
                return {"data": decoded}
        except (json.JSONDecodeError, RecursionError):
            pass

        return {}

    def _extract_from_file_path(self, file_path: str) -> Dict[str, Any]:
        """Extract ComfyUI workflow data from a PNG file path."""
        import os

        if not file_path or not os.path.exists(file_path):
            print(f"MetaComb: File path does not exist: {file_path}")
            return {}

        try:
            with Image.open(file_path) as img:
                result = self._extract_workflow_from_png(img)
                if result:
                    print(
                        "MetaComb: Successfully extracted metadata from file:",
                        file_path,
                    )
                else:
                    print(
                        "MetaComb: No metadata found in file:",
                        file_path,
                    )
                    print(
                        "MetaComb: Available PNG info keys:",
                        list(img.info.keys()),
                    )
                return result
        except Exception as e:
            print("MetaComb: Error reading file", file_path, ":", e)
            return {}

    def _extract_from_image(self, image: Any) -> Dict[str, Any]:
        """Extract ComfyUI workflow data from image tensor or PIL Image."""

        # Convert tensor to PIL Image if needed
        if hasattr(image, "shape"):
            try:
                import torch
                import numpy as np

                # Validate it's a torch tensor
                if isinstance(image, torch.Tensor):
                    tensor_data = image[0].cpu().numpy()  # type: ignore
                    i = 255.0 * tensor_data
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                else:
                    print(
                        "MetaComb: Image has shape but is not torch.Tensor:",
                        type(image),
                    )
                    return {}
            except Exception as e:
                print(f"MetaComb: Error converting tensor to PIL: {e}")
                return {}
        else:
            img = image

        result = self._extract_workflow_from_png(img)
        if not result:
            print(
                f"MetaComb: No metadata extracted from image. "
                f"Image type: {type(img)}"
            )
            if isinstance(img, Image.Image):
                print(f"MetaComb: PIL Image metadata keys: {list(img.info.keys())}")
        return result

    def _extract_workflow_from_png(
        self, img: Image.Image
    ) -> Dict[str, Any]:
        """Extract ComfyUI workflow data from PNG metadata."""
        if not isinstance(img, Image.Image):  # type: ignore
            return {}

        metadata = img.info
        workflow_data: Dict[str, Any] = {}

        # Extract both workflow and prompt keys if they exist
        for key in ["workflow", "prompt"]:
            if key in metadata:
                try:
                    data = (
                        json.loads(metadata[key])
                        if isinstance(metadata[key], str)
                        else metadata[key]
                    )
                    workflow_data[key] = data
                except json.JSONDecodeError:
                    continue

        # If we found workflow or prompt data, return it
        if workflow_data:
            return workflow_data

        # Try to reconstruct from separate keys as fallback
        for meta_key, meta_value in metadata.items():
            # Ensure meta_key is a string (PIL can have tuple keys)
            if not isinstance(meta_key, str):
                continue

            is_json_str = (
                isinstance(meta_value, str)
                and meta_value.strip().startswith("{")
            )
            if is_json_str:
                try:
                    parsed = json.loads(meta_value)
                    is_workflow_key = meta_key in ["workflow", "prompt"]
                    has_class_type = "class_type" in str(parsed)
                    if is_workflow_key or has_class_type:
                        workflow_data[meta_key] = parsed
                except json.JSONDecodeError:
                    continue

        return workflow_data if workflow_data else {}

    def _search_nodes(
        self,
        data: Dict[str, Any],
        key: str,
        node_title: str = "",
        node_type: str = "",
    ) -> List[Any]:
        """Search nodes based on filters and extract key values."""

        if not isinstance(data, dict):  # type: ignore
            return []

        results: List[Any] = []

        # If key is not provided, return node objects based on title/type
        if not (key and str(key).strip()):
            if node_title and node_type:
                return self._nodes_by_title_and_type(data, node_title, node_type)
            if node_title:
                return self._nodes_by_title(data, node_title)
            if node_type:
                return self._nodes_by_type(data, node_type)

        # Handle both node_title and node_type specified (search for key)
        if node_title and node_type:
            results = self._search_by_title_and_type(
                data, key, node_title, node_type
            )

        # Handle only node_title specified
        elif node_title:
            results = self._search_by_title(data, key, node_title)

        # Handle only node_type specified
        elif node_type:
            results = self._search_by_type(data, key, node_type)

        # No filters - search all nodes
        else:
            results = self._search_all_nodes(data, key)

        return results

    def _search_by_title_and_type(
        self,
        data: Dict[str, Any],
        key: str,
        node_title: str,
        node_type: str,
    ) -> List[Any]:
        """Search for key in nodes matching both title and type."""
        results: List[Any] = []

        for _node_id, node_data in data.items():
            if not isinstance(node_data, dict):
                continue

            # Check if node matches both type and title
            class_type_val: Any = node_data.get("class_type")  # type: ignore
            if class_type_val == node_type:
                meta_val: Any = node_data.get("_meta", {})  # type: ignore
                meta_dict = cast(Dict[str, Any], meta_val)
                title_val: Any = meta_dict.get("title", "")  # type: ignore
                meta_title = str(title_val)
                if meta_title == node_title:
                    value = self._recursive_find_key(node_data, key)
                    if value is not None:
                        results.append(value)

        return results

    def _search_by_title(
        self, data: Dict[str, Any], key: str, node_title: str
    ) -> List[Any]:
        """Search for key in nodes matching title."""
        matching_nodes: List[Any] = []

        for _node_id, node_data in data.items():
            if not isinstance(node_data, dict):
                continue

            meta_val: Any = node_data.get("_meta", {})  # type: ignore
            meta_dict = cast(Dict[str, Any], meta_val)
            title_val: Any = meta_dict.get("title", "")  # type: ignore
            meta_title = str(title_val)
            if meta_title == node_title:
                value = self._recursive_find_key(node_data, key)
                if value is not None:
                    matching_nodes.append(value)

        # Return all matches if multiple found, otherwise first match
        if len(matching_nodes) > 1:
            # Check if all values are the same
            if len(set(str(v) for v in matching_nodes)) == 1:
                return [matching_nodes[0]]
            return matching_nodes

        return matching_nodes

    def _search_by_type(
        self, data: Dict[str, Any], key: str, node_type: str
    ) -> List[Any]:
        """Search for key in all nodes of specified type."""
        results: List[Any] = []

        for _node_id, node_data in data.items():
            if not isinstance(node_data, dict):
                continue

            class_type_val: Any = node_data.get("class_type")  # type: ignore
            if class_type_val == node_type:
                value = self._recursive_find_key(node_data, key)
                if value is not None:
                    results.append(value)

        return results

    def _search_all_nodes(
        self, data: Dict[str, Any], key: str
    ) -> List[Any]:
        """Search for key in all nodes."""
        results: List[Any] = []

        for _node_id, node_data in data.items():
            if not isinstance(node_data, dict):
                continue

            value = self._recursive_find_key(node_data, key)
            if value is not None:
                results.append(value)
                return results  # Return first match

        return results

    def _nodes_by_title_and_type(
        self,
        data: Dict[str, Any],
        node_title: str,
        node_type: str,
    ) -> List[Dict[str, Any]]:
        """Return node objects matching both title and type."""
        results: List[Dict[str, Any]] = []
        for _node_id, node_data in data.items():
            if not isinstance(node_data, dict):
                continue
            class_type_val: Any = node_data.get("class_type")  # type: ignore
            if class_type_val == node_type:
                meta_val: Any = node_data.get("_meta", {})  # type: ignore
                meta_dict = cast(Dict[str, Any], meta_val)
                title_val: Any = meta_dict.get("title", "")  # type: ignore
                if str(title_val) == node_title:
                    results.append(node_data)
        return results

    def _nodes_by_title(self, data: Dict[str, Any], node_title: str) -> List[Dict[str, Any]]:
        """Return node objects matching title."""
        results: List[Dict[str, Any]] = []
        for _node_id, node_data in data.items():
            if not isinstance(node_data, dict):
                continue
            meta_val: Any = node_data.get("_meta", {})  # type: ignore
            meta_dict = cast(Dict[str, Any], meta_val)
            title_val: Any = meta_dict.get("title", "")  # type: ignore
            if str(title_val) == node_title:
                results.append(node_data)
        return results

    def _nodes_by_type(self, data: Dict[str, Any], node_type: str) -> List[Dict[str, Any]]:
        """Return node objects matching type."""
        results: List[Dict[str, Any]] = []
        for _node_id, node_data in data.items():
            if not isinstance(node_data, dict):
                continue
            class_type_val: Any = node_data.get("class_type")  # type: ignore
            if class_type_val == node_type:
                results.append(node_data)
        return results

    def _recursive_find_key(self, obj: Any, key: str) -> Any:
        """Recursively search for key in nested structure."""

        if isinstance(obj, dict):
            # Direct key match
            if key in obj:
                return cast(Any, obj[key])

            # Recursive search in nested dicts
            for value in obj.values():  # type: ignore
                typed_value: Any = cast(Any, value)
                result = self._recursive_find_key(typed_value, key)
                if result is not None:
                    return result

        elif isinstance(obj, list):
            # Search in list items
            for item in obj:  # type: ignore
                typed_item: Any = cast(Any, item)
                result = self._recursive_find_key(typed_item, key)
                if result is not None:
                    return result

        return None


NODE_CLASS_MAPPINGS = {"MetaComb": MetaComb}

NODE_DISPLAY_NAME_MAPPINGS = {"MetaComb": "Meta Comb"}
