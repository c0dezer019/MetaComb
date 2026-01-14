"""Load Image node with filepath output for use with MetaComb."""
import hashlib
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

# Import ComfyUI utilities
import folder_paths  # type: ignore
import node_helpers  # type: ignore


class LoadImageWithPath:
    """Load Image node that outputs filepath, image and mask."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            }
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filepath")
    FUNCTION = "load_image"

    def load_image(self, image: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Load image and return image tensor, mask, and filepath.

        Args:
            image: Image filename to load

        Returns:
            Tuple of (image tensor, mask tensor, filepath string)
        """
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image_rgb = i.convert("RGB")

            if len(output_images) == 0:
                w = image_rgb.size[0]
                h = image_rgb.size[1]

            if image_rgb.size[0] != w or image_rgb.size[1] != h:
                continue

            image_array = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]

            # Handle alpha channel / transparency
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            elif i.mode == "P" and "transparency" in i.info:
                mask_array = np.array(
                    i.convert("RGBA").getchannel("A")
                ).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_array)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))

            if img.format == "MPO":
                break  # ignore all frames except the first one for MPO format

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, image_path)

    @classmethod
    def IS_CHANGED(cls, image: str) -> str:
        """Return hash of image file to detect changes."""
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image: str) -> bool | str:
        """Validate that the image file exists."""
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {"LoadImageWithPath": LoadImageWithPath}

NODE_DISPLAY_NAME_MAPPINGS = {"LoadImageWithPath": "Load Image (with Path)"}
