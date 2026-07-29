"""MDC ComfyUI nodes."""

from .mdc_face_crop import MdcFaceCrop

NODE_CLASS_MAPPINGS = {
    "MdcFaceCrop": MdcFaceCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MdcFaceCrop": "MDC Face Crop (YuNet)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
