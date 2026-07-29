"""ComfyUI node: crop the face out of an image with YuNet (OpenCV, CPU).

Replaces the UltralyticsDetectorProvider -> BboxDetectorSEGS ->
SEGSOrderedFilter -> SEGSToImageList chain with a single node.

Why: the Ultralytics chain keeps a YOLO model with CUDA tensors alive in the
ComfyUI process. That model is not a `ModelPatcher`, so ComfyUI's memory
manager does not track it — under VRAM pressure its weights get freed or
overwritten and the detector silently starts returning boxes over random
background instead of faces (no exception, no log). YuNet runs inside OpenCV on
the CPU, so there is no CUDA memory for anything to reclaim.
"""

from __future__ import annotations

import logging
from typing import Tuple

from .face_detect import crop_face

logger = logging.getLogger(__name__)

_SELECTION_STRATEGIES = ["confidence", "largest"]
_FALLBACKS = ["full_image", "error"]


class MdcFaceCrop:
    """Detects one face and returns it as a cropped IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": "Box expansion around the detected face. 1.0 = the bare face box.",
                    },
                ),
                "conf_threshold": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.05, "max": 1.0, "step": 0.01},
                ),
                "selection": (
                    _SELECTION_STRATEGIES,
                    {
                        "default": "confidence",
                        "tooltip": "Which detection to keep. 'largest' lets one oversized false positive win — prefer 'confidence'.",
                    },
                ),
                "fallback": (
                    _FALLBACKS,
                    {
                        "default": "full_image",
                        "tooltip": "No face found: pass the full image through (degrade) or raise (fail loudly).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "face_found", "info")
    FUNCTION = "crop"
    CATEGORY = "MDC/face"
    DESCRIPTION = "Crop the face from an image using YuNet (OpenCV, CPU — no CUDA state to corrupt)."

    def crop(
        self,
        image,
        crop_factor: float,
        conf_threshold: float,
        selection: str,
        fallback: str,
    ) -> Tuple[object, bool, str]:
        import numpy as np
        import torch

        crops = []
        infos = []
        found_any = False

        for index in range(image.shape[0]):
            frame = image[index]
            rgb = (frame.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            bgr = rgb[:, :, ::-1].copy()

            cropped_bgr, face = crop_face(
                bgr,
                crop_factor=crop_factor,
                conf_threshold=conf_threshold,
                strategy=selection,
            )

            if face is None:
                message = (
                    f"no face found (threshold {conf_threshold}) in "
                    f"{bgr.shape[1]}x{bgr.shape[0]}"
                )
                if fallback == "error":
                    raise RuntimeError(f"MdcFaceCrop: {message}")
                logger.warning("MdcFaceCrop: %s; passing the full image through", message)
                infos.append(message)
                crops.append(frame.unsqueeze(0))
                continue

            found_any = True
            height, width = cropped_bgr.shape[:2]
            message = (
                f"face {face.box[2]}x{face.box[3]} conf={face.confidence:.3f} "
                f"-> crop {width}x{height}"
            )
            logger.info("MdcFaceCrop: %s", message)
            infos.append(message)

            cropped_rgb = cropped_bgr[:, :, ::-1].copy()
            tensor = torch.from_numpy(cropped_rgb.astype(np.float32) / 255.0)
            crops.append(tensor.unsqueeze(0))

        # Crops of one batch can differ in size, and an IMAGE output must be a
        # single stacked tensor. Our callers send one image at a time; if a real
        # batch ever arrives with mismatched crops, keep the first and say so.
        shapes = {tuple(c.shape[1:]) for c in crops}
        if len(shapes) > 1:
            logger.warning(
                "MdcFaceCrop: batch produced %d different crop sizes; keeping the first",
                len(shapes),
            )
            crops = crops[:1]
            infos = infos[:1]

        return torch.cat(crops, dim=0), found_any, " | ".join(infos)
