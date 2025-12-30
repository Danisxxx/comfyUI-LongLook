"""
comfyUI-LongLook: FreeLong for Wan 2.2 Video Generation

Implements FreeLong (NeurIPS 2024) spectral blending for:
1. Better motion consistency within each generation
2. Reliable chunk chaining for unlimited length videos
"""

from .nodes import (
    WanContinuationConditioning,
    WanFreeLong,
    WanFreeLongEnforcer,
    WanMotionScale,
    WanMotionScaleAdvanced,
)

NODE_CLASS_MAPPINGS = {
    "WanContinuationConditioning": WanContinuationConditioning,
    "WanFreeLong": WanFreeLong,
    "WanFreeLongEnforcer": WanFreeLongEnforcer,
    "WanMotionScale": WanMotionScale,
    "WanMotionScaleAdvanced": WanMotionScaleAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanContinuationConditioning": "Wan Continuation Conditioning",
    "WanFreeLong": "Wan FreeLong",
    "WanFreeLongEnforcer": "Wan FreeLong Enforcer",
    "WanMotionScale": "Wan Motion Scale (Experimental)",
    "WanMotionScaleAdvanced": "Wan Motion Scale Advanced (Experimental)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None
__version__ = "3.0.6"
