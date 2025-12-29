import torch
import math
import copy
import logging
from typing import Tuple, Dict, Any, List, Optional

import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.latent_formats
import latent_preview
import node_helpers

# Set up logging for FreeLong
logger = logging.getLogger("comfyui.longlook")
logger.setLevel(logging.INFO)


class WanContinuationConditioning:
    """
    Simple video continuation conditioning for Wan 2.2.

    Takes the last frame from a previous video chunk and creates i2v conditioning
    for generating the next chunk. Equivalent to WanImageToVideo but designed
    for easy chaining in continuation workflows.

    Usage: Connect decoded images from chunk 1 → this node → KSampler for chunk 2
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "anchor_images": ("IMAGE",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "video_length": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 1024,
                    "step": 4,
                    "tooltip": "Output video length in frames. Must match your sampler settings."
                }),
            },
            "optional": {
                "end_images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "modify_conditioning"
    CATEGORY = "video/wan"
    DESCRIPTION = "Creates i2v conditioning from last frame with optional end-frame guidance"

    def modify_conditioning(
        self,
        positive,
        negative,
        anchor_images: torch.Tensor,
        vae,
        width: int,
        height: int,
        video_length: int,
        end_images: Optional[torch.Tensor] = None
    ) -> Tuple[list, list, Dict[str, torch.Tensor]]:
        """
        Modify conditioning to use anchor frames for continuation.
        """
        import comfy.utils

        # Calculate latent dimensions using Wan's formula
        latent_frames = ((video_length - 1) // 4) + 1
        latent_h = height // 8
        latent_w = width // 8

        # Get the LAST frame from anchor images (single frame continuation like standard i2v)
        start_frame = anchor_images[-1:]  # Always use just the last frame

        end_frame = None
        if end_images is not None:
            end_frame = end_images[-1:]  # Use last frame as end anchor

        # Resize to target dimensions if needed
        if start_frame.shape[1] != height or start_frame.shape[2] != width:
            start_frame = comfy.utils.common_upscale(
                start_frame.movedim(-1, 1),
                width, height, "bilinear", "center"
            ).movedim(1, -1)

        if end_frame is not None and (end_frame.shape[1] != height or end_frame.shape[2] != width):
            end_frame = comfy.utils.common_upscale(
                end_frame.movedim(-1, 1),
                width, height, "bilinear", "center"
            ).movedim(1, -1)
        if end_frame is not None and (end_frame.device != start_frame.device or end_frame.dtype != start_frame.dtype):
            end_frame = end_frame.to(device=start_frame.device, dtype=start_frame.dtype)

        # Create full video tensor for VAE encoding (matches WanImageToVideo exactly)
        # Gray (0.5) for frames to generate, actual image for first frame
        full_video = torch.ones(
            video_length, height, width, 3,
            device=start_frame.device, dtype=start_frame.dtype
        ) * 0.5

        # Place the start anchor at the beginning
        full_video[:1] = start_frame[:, :, :, :3]

        # Optionally place end anchor at the end
        if end_frame is not None:
            full_video[-1:] = end_frame[:, :, :, :3]

        # VAE encode the entire video (anchor frames + gray neutral frames)
        concat_latent_image = vae.encode(full_video)
        if concat_latent_image.ndim == 4:
            concat_latent_image = concat_latent_image.unsqueeze(0)

        encoded_latent_frames = concat_latent_image.shape[2]
        latent_channels = concat_latent_image.shape[1]

        # Calculate how many latent frames to preserve (1 video frame = 1 latent frame)
        # Formula: ((1 - 1) // 4) + 1 = 1
        anchor_latent_frames = 1

        if end_frame is None:
            # Create concat_mask using LATENT frames (like WanImageToVideo does - simpler, no reshape)
            # Shape: [1, 1, latent_frames, h, w]
            # Wan convention: 0.0 = preserve (don't denoise), 1.0 = generate (full denoise)
            concat_mask = torch.ones(
                1, 1, encoded_latent_frames,
                concat_latent_image.shape[-2],
                concat_latent_image.shape[-1],
                device=start_frame.device, dtype=start_frame.dtype
            )
            # Preserve anchor latent frames
            concat_mask[:, :, :anchor_latent_frames] = 0.0
        else:
            # Core-style frame mask with end-frame preservation, then reshape to latent mask
            frame_mask = torch.ones(
                1, 1, encoded_latent_frames * 4,
                concat_latent_image.shape[-2],
                concat_latent_image.shape[-1],
                device=start_frame.device, dtype=start_frame.dtype
            )
            frame_mask[:, :, :start_frame.shape[0] + 3] = 0.0
            frame_mask[:, :, -end_frame.shape[0]:] = 0.0
            concat_mask = frame_mask.view(
                1, frame_mask.shape[2] // 4, 4,
                frame_mask.shape[-2], frame_mask.shape[-1]
            ).movedim(1, 2)

        # Move to CPU for conditioning storage
        concat_latent_image = concat_latent_image.cpu()
        concat_mask = concat_mask.cpu()

        # Modify conditioning with our anchor data
        # This overwrites any existing concat_latent_image/concat_mask from WanImageToVideo
        positive_out = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": concat_mask
        })
        negative_out = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": concat_mask
        })

        # Create empty latent for sampling
        latent = torch.zeros(
            1, latent_channels, latent_frames, latent_h, latent_w,
            device=comfy.model_management.intermediate_device()
        )

        return (positive_out, negative_out, {"samples": latent})


# ============================================================================
# FreeLong Helper Functions
# ============================================================================

def freelong_enforced_blend(
    global_features: torch.Tensor,
    local_features: torch.Tensor,
    motion_lock_ratio: float = 0.15,
    low_freq_ratio: float = 0.5,
    blend_strength: float = 1.0
) -> torch.Tensor:
    """
    Enforced spectral blend with 3-tier frequency handling:
    - Ultra-low (0 to motion_lock_ratio): 100% global - LOCKED motion skeleton
    - Mid (motion_lock_ratio to low_freq_ratio): Blended global/local
    - High (low_freq_ratio to 1.0): 100% local - fine details

    This ensures motion trajectory cannot be overridden by local attention.
    """
    if blend_strength == 0.0:
        return local_features

    orig_dtype = global_features.dtype
    global_float = global_features.float()
    local_float = local_features.float()

    # Single FFT pass
    global_freq = torch.fft.rfft(global_float, dim=1)
    local_freq = torch.fft.rfft(local_float, dim=1)
    del global_float, local_float

    num_freq = global_freq.shape[1]
    lock_cutoff = int(num_freq * motion_lock_ratio)
    blend_cutoff = int(num_freq * low_freq_ratio)

    # Build 3-tier mask: 1.0 = global, 0.0 = local
    global_mask = torch.zeros(num_freq, device=global_features.device, dtype=torch.float32)

    # Tier 1: Ultra-low frequencies - 100% global (locked)
    global_mask[:lock_cutoff] = 1.0

    # Tier 2: Mid frequencies - smooth transition from global to local
    if blend_cutoff > lock_cutoff:
        mid_len = blend_cutoff - lock_cutoff
        transition = torch.linspace(1, 0, mid_len + 2, device=global_features.device, dtype=torch.float32)[1:-1]
        global_mask[lock_cutoff:blend_cutoff] = transition
        del transition

    # Tier 3: High frequencies - 100% local (already 0.0)

    global_mask = global_mask.view(1, -1, *([1] * (len(global_freq.shape) - 2)))

    # Blend: global * mask + local * (1 - mask)
    blended_freq = global_freq * global_mask + local_freq * (1.0 - global_mask)
    del global_freq, local_freq, global_mask

    result = torch.fft.irfft(blended_freq, n=global_features.shape[1], dim=1)
    del blended_freq

    if blend_strength < 1.0:
        result = local_features.float() * (1.0 - blend_strength) + result * blend_strength

    return result.to(orig_dtype)


def freelong_spectral_blend(
    global_features: torch.Tensor,
    local_features: torch.Tensor,
    low_freq_ratio: float = 0.25,
    blend_strength: float = 1.0
) -> torch.Tensor:
    """
    Blend global and local features using spectral decomposition.
    Low frequencies from global (structure/motion direction),
    High frequencies from local (details/sharpness).

    Memory-optimized: Uses single FFT pass instead of two separate filter calls.

    Args:
        global_features: Features from full-sequence attention
        local_features: Features from windowed attention
        low_freq_ratio: Ratio of frequencies to take from global (default 0.25)
        blend_strength: How much to apply the blend (0=passthrough, 1=full blend)

    Returns:
        Blended features
    """
    if blend_strength == 0.0:
        return local_features

    # Convert to float32 for FFT (cuFFT doesn't support half precision for non-power-of-two)
    orig_dtype = global_features.dtype
    global_float = global_features.float()
    local_float = local_features.float()

    # Single FFT pass for both streams
    global_freq = torch.fft.rfft(global_float, dim=1)
    local_freq = torch.fft.rfft(local_float, dim=1)
    del global_float, local_float  # Free immediately

    # Build low-pass mask
    num_freq = global_freq.shape[1]
    cutoff = int(num_freq * low_freq_ratio)
    transition_width = max(1, min(num_freq // 4, cutoff // 2))

    low_mask = torch.ones(num_freq, device=global_features.device, dtype=torch.float32)
    transition_start = max(0, cutoff - transition_width // 2)
    transition_end = min(num_freq, cutoff + transition_width // 2)

    if transition_end > transition_start:
        transition_idx = torch.arange(transition_start, transition_end, device=global_features.device, dtype=torch.float32)
        transition_values = 0.5 * (1 + torch.cos(torch.pi * (transition_idx - transition_start) / (transition_end - transition_start)))
        low_mask[transition_start:transition_end] = transition_values
        del transition_idx, transition_values

    low_mask[transition_end:] = 0.0
    low_mask = low_mask.view(1, -1, *([1] * (len(global_freq.shape) - 2)))

    # Blend in frequency domain: low from global, high from local
    # high_mask = 1 - low_mask, so: global*low + local*high = global*low + local*(1-low)
    blended_freq = global_freq * low_mask + local_freq * (1.0 - low_mask)
    del global_freq, local_freq, low_mask  # Free immediately

    # Inverse FFT
    result = torch.fft.irfft(blended_freq, n=global_features.shape[1], dim=1)
    del blended_freq

    # Mix with original based on blend_strength
    if blend_strength < 1.0:
        result = local_features.float() * (1.0 - blend_strength) + result * blend_strength

    return result.to(orig_dtype)


# ============================================================================
# FreeLong Model Patcher
# ============================================================================

class WanFreeLong:
    """
    FreeLong model patcher for Wan 2.2 video generation.

    Implements SpectralBlend temporal attention to extend video generation
    beyond the training context window while maintaining motion consistency.

    Based on "FreeLong: Training-Free Long Video Generation with
    SpectralBlend Temporal Attention" (NeurIPS 2024)

    How it works:
    1. Hooks into Wan's transformer blocks
    2. Runs attention twice: global (full sequence) and local (windowed)
    3. Blends results using FFT: low-freq from global, high-freq from local
    4. This preserves motion direction (low-freq) while keeping details (high-freq)
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Turn FreeLong on or off. Useful for comparing results with and without."
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "How strongly FreeLong affects the video. Higher = more consistent motion but slightly softer. Lower = sharper but may drift. 0.8 is optimal for most cases."
                }),
                "low_freq_ratio": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Balance between motion consistency and fine detail. Higher = smoother, more consistent motion. Lower = more dynamic but may drift. 0.8 is a good balance."
                }),
                "local_window_frames": ("INT", {
                    "default": 33,
                    "min": 9,
                    "max": 241,
                    "step": 4,
                    "tooltip": "How many video frames the detail-preservation looks at. Smaller = sharper details but may cause morphing. Larger = smoother transitions. 33 works well for 81-frame videos. For longer videos, try ~40% of total frames."
                }),
                "blend_start_block": ("INT", {
                    "default": 0,
                    "min": -1,
                    "max": 40,
                    "tooltip": "Advanced: Which model layer to start applying FreeLong. 0 = from the beginning. Most users should leave this at 0."
                }),
                "blend_end_block": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 40,
                    "tooltip": "Advanced: Which model layer to stop applying FreeLong. -1 = apply to all layers. Most users should leave this at -1."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "video/wan"
    DESCRIPTION = "Apply FreeLong spectral blending for better motion consistency in long videos"

    def patch_model(
        self,
        model,
        enabled: bool,
        blend_strength: float,
        low_freq_ratio: float,
        local_window_frames: int,
        blend_start_block: int,
        blend_end_block: int
    ):
        """
        Patch the model to use FreeLong spectral blending in temporal attention.
        """
        # Convert video frames to latent frames
        local_window_size = ((local_window_frames - 1) // 4) + 1
        # If disabled, return model unchanged (no patches)
        if not enabled:
            logger.info("[FreeLong] DISABLED - Model returned without patching")
            return (model,)

        logger.info("=" * 60)
        logger.info("[FreeLong] APPLYING SPECTRAL BLENDING PATCHES")
        logger.info("=" * 60)
        logger.info(f"  Blend Strength: {blend_strength:.2f}")
        logger.info(f"  Low Freq Ratio: {low_freq_ratio:.2f}")
        logger.info(f"  Local Window: {local_window_frames} video frames → {local_window_size} latent frames")
        logger.info(f"  Block Range: {blend_start_block} to {blend_end_block if blend_end_block != -1 else 'ALL'}")

        model = model.clone()

        # Store settings for the patch function
        freelong_settings = {
            "blend_strength": blend_strength,
            "low_freq_ratio": low_freq_ratio,
            "local_window_size": local_window_size,
            "blend_start_block": blend_start_block,
            "blend_end_block": blend_end_block,
            "first_run": True,  # Track first execution to log details once
            "blocks_patched": set(),  # Track which blocks executed
        }

        def freelong_block_patch(block_args, extra_args):
            """
            Patch function that wraps transformer blocks with FreeLong blending.

            Implements dual-stream processing:
            1. Global stream: Full sequence attention (original behavior)
            2. Local stream: Windowed attention (process temporal chunks)
            3. Spectral blend: Low-freq from global, high-freq from local
            """
            original_block = extra_args["original_block"]
            transformer_options = block_args.get("transformer_options", {})
            block_index = transformer_options.get("block_index", 0)
            total_blocks = transformer_options.get("total_blocks", 40)

            # Determine if this block should use FreeLong
            start = freelong_settings["blend_start_block"]
            end = freelong_settings["blend_end_block"]
            if end == -1:
                end = total_blocks

            # For blocks outside our range, just run original
            if block_index < start or block_index >= end:
                return original_block(block_args)

            # Skip if blend_strength is 0
            strength = freelong_settings["blend_strength"]
            if strength == 0.0:
                return original_block(block_args)

            # Track that this block executed (for logging summary)
            freelong_settings["blocks_patched"].add(block_index)

            try:
                x = block_args["img"]  # [batch, seq_len, dim]
                batch_size, seq_len, hidden_dim = x.shape

                window_size = freelong_settings["local_window_size"]
                low_ratio = freelong_settings["low_freq_ratio"]

                # Log details on first execution
                if freelong_settings["first_run"]:
                    freelong_settings["first_run"] = False
                    logger.info("")
                    logger.info("[FreeLong] DUAL-STREAM PROCESSING STARTING")
                    logger.info(f"  Input shape: batch={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
                    logger.info(f"  Window size: {window_size} latent frames")

                # ============ GLOBAL STREAM ============
                # Full sequence attention - preserves global motion patterns
                global_out = original_block(block_args)
                global_x = global_out["img"]

                # ============ LOCAL STREAM ============
                # Process sequence in temporal windows for local detail preservation
                #
                # Wan's sequence is flattened [frames * spatial, dim]
                # We need to estimate the temporal structure
                # For 81 frame video: 21 latent frames, so spatial = seq_len / 21

                # Use cached values if available (avoids repeated detection across 40 blocks)
                cached_seq_len = freelong_settings.get("cached_seq_len")
                if cached_seq_len == seq_len:
                    # Use cached values
                    num_frames = freelong_settings["cached_num_frames"]
                    spatial_size = freelong_settings["cached_spatial_size"]
                    freqs_seq_dim = freelong_settings.get("cached_freqs_seq_dim")
                    overlap = freelong_settings["cached_overlap"]
                    stride = freelong_settings["cached_stride"]
                else:
                    # First time or seq_len changed - detect and cache
                    # Try to infer frame count from sequence length
                    # Common Wan resolutions and their spatial token counts:
                    # 1280x720 -> 160x90/4 = 40x22.5 -> ~900 tokens per frame (after 2x2 patch)
                    # 848x480 -> 106x60/4 = 26.5x15 -> ~400 tokens per frame
                    # We'll try common latent frame counts: 21, 17, 13, 9, 5

                    num_frames = None
                    for candidate_frames in [21, 17, 13, 9, 5]:
                        if seq_len % candidate_frames == 0:
                            num_frames = candidate_frames
                            break

                    if num_frames is None or num_frames <= window_size:
                        # Can't determine temporal structure or window covers all frames
                        # Fall back to just using global output with spectral smoothing
                        if freelong_settings.get("log_fallback", True):
                            freelong_settings["log_fallback"] = False
                            if num_frames is None:
                                logger.warning(f"[FreeLong] Could not determine temporal structure (seq_len={seq_len})")
                            else:
                                logger.warning(f"[FreeLong] Window size >= frames ({window_size} >= {num_frames})")
                            logger.warning("[FreeLong] Using fallback: spectral smoothing without windowing")
                        blended_x = freelong_spectral_blend(global_x, global_x, low_ratio, strength)
                        return {"img": blended_x}

                    spatial_size = seq_len // num_frames
                    overlap = window_size // 2
                    stride = window_size - overlap

                    # Detect RoPE sequence dimension
                    freqs = block_args.get("pe")
                    freqs_seq_dim = None
                    if freqs is not None:
                        for dim_idx, dim_size in enumerate(freqs.shape):
                            if dim_size == seq_len:
                                freqs_seq_dim = dim_idx
                                break

                    # Cache all computed values
                    freelong_settings["cached_seq_len"] = seq_len
                    freelong_settings["cached_num_frames"] = num_frames
                    freelong_settings["cached_spatial_size"] = spatial_size
                    freelong_settings["cached_freqs_seq_dim"] = freqs_seq_dim
                    freelong_settings["cached_overlap"] = overlap
                    freelong_settings["cached_stride"] = stride

                    # Precompute blend ramps (avoid torch.linspace per window)
                    if overlap > 0:
                        ramp_up = torch.linspace(0, 1, overlap + 1, device=x.device, dtype=x.dtype)[1:]
                        ramp_down = torch.linspace(1, 0, overlap + 1, device=x.device, dtype=x.dtype)[:-1]
                        freelong_settings["cached_ramp_up"] = ramp_up
                        freelong_settings["cached_ramp_down"] = ramp_down

                    # Log temporal structure detection on first run
                    if freelong_settings.get("log_structure", True):
                        freelong_settings["log_structure"] = False
                        logger.info(f"  Detected temporal structure: {num_frames} frames × {spatial_size} spatial tokens")

                    # Calculate and log number of windows
                    num_windows = 0
                    temp_start = 0
                    while temp_start < num_frames:
                        num_windows += 1
                        temp_start += stride
                        if temp_start >= num_frames:
                            break

                    if freelong_settings.get("log_windows", True):
                        freelong_settings["log_windows"] = False
                        logger.info(f"  Window processing: {num_windows} windows with {overlap} frame overlap")
                        logger.info(f"  Overlap strategy: 50% crossfade blending between windows")

                # Get RoPE embeddings
                freqs = block_args.get("pe")

                # Reshape to [batch, frames, spatial, dim]
                x_temporal = x.view(batch_size, num_frames, spatial_size, hidden_dim)

                # Accumulator for blended output and weights
                local_x_temporal = torch.zeros(batch_size, num_frames, spatial_size, hidden_dim,
                                               device=x.device, dtype=x.dtype)
                weight_accumulator = torch.zeros(batch_size, num_frames, 1, 1,
                                                  device=x.device, dtype=x.dtype)

                # Get cached ramps if available
                cached_ramp_up = freelong_settings.get("cached_ramp_up")
                cached_ramp_down = freelong_settings.get("cached_ramp_down")

                # Save original values for in-place modification (avoid dict copy per window)
                original_img = block_args.get("img")
                original_pe = block_args.get("pe")

                # Process overlapping windows
                start_frame = 0
                while start_frame < num_frames:
                    end_frame = min(start_frame + window_size, num_frames)
                    window_frames = end_frame - start_frame

                    # Extract window and flatten for block processing
                    window_x = x_temporal[:, start_frame:end_frame, :, :]
                    window_flat = window_x.reshape(batch_size, window_frames * spatial_size, hidden_dim)

                    # Modify block_args in-place (restore after)
                    block_args["img"] = window_flat

                    # Slice RoPE embeddings to match window
                    if freqs is not None and freqs_seq_dim is not None:
                        start_token = start_frame * spatial_size
                        end_token = end_frame * spatial_size
                        slices = [slice(None)] * len(freqs.shape)
                        slices[freqs_seq_dim] = slice(start_token, end_token)
                        block_args["pe"] = freqs[tuple(slices)]

                    # Run block on window
                    window_out = original_block(block_args)
                    window_result = window_out["img"]

                    # Reshape back to temporal
                    window_result = window_result.view(batch_size, window_frames, spatial_size, hidden_dim)

                    # Create blending weights using cached ramps
                    # This creates smooth crossfades in overlapping regions
                    weights = torch.ones(window_frames, device=x.device, dtype=x.dtype)

                    if start_frame > 0 and overlap > 0 and cached_ramp_up is not None:
                        # Ramp up at the start (we're overlapping with previous window)
                        ramp_len = min(overlap, window_frames)
                        weights[:ramp_len] = cached_ramp_up[:ramp_len]

                    if end_frame < num_frames and overlap > 0 and cached_ramp_down is not None:
                        # Ramp down at the end (next window will overlap with us)
                        ramp_len = min(overlap, window_frames)
                        weights[-ramp_len:] = weights[-ramp_len:] * cached_ramp_down[:ramp_len]

                    weights = weights.view(1, window_frames, 1, 1)

                    # Accumulate weighted results (in-place for memory efficiency)
                    weighted_result = window_result * weights
                    local_x_temporal[:, start_frame:end_frame].add_(weighted_result)
                    weight_accumulator[:, start_frame:end_frame].add_(weights)

                    # Free window tensors immediately
                    del window_x, window_flat, window_out
                    del window_result, weights, weighted_result

                    # Move to next window position
                    start_frame += stride
                    if start_frame >= num_frames:
                        break

                # Restore original block_args
                block_args["img"] = original_img
                if original_pe is not None:
                    block_args["pe"] = original_pe

                # Normalize by accumulated weights
                weight_accumulator = weight_accumulator.clamp(min=1e-8)
                local_x_temporal.div_(weight_accumulator)  # In-place division
                del weight_accumulator
                local_x = local_x_temporal.reshape(batch_size, seq_len, hidden_dim)
                del local_x_temporal, x_temporal

                # ============ SPECTRAL BLEND ============
                # Low frequencies from global (motion direction/consistency)
                # High frequencies from local (sharp details)
                if freelong_settings.get("log_blend", True):
                    freelong_settings["log_blend"] = False
                    logger.info("")
                    logger.info("[FreeLong] SPECTRAL BLENDING")
                    logger.info(f"  Low-freq (global): {low_ratio:.0%} of spectrum (motion consistency)")
                    logger.info(f"  High-freq (local): {1-low_ratio:.0%} of spectrum (sharp details)")
                    logger.info(f"  Blend strength: {strength:.0%}")

                blended_x = freelong_spectral_blend(global_x, local_x, low_ratio, strength)

                # Free large intermediate tensors
                del global_x, local_x

                return {"img": blended_x}

            except Exception as e:
                # If anything fails, fall back to original block
                logger.error(f"[FreeLong] Error in block {block_index}: {e}", exc_info=True)
                return original_block(block_args)

        # Wan 2.2 typically has 40 blocks
        num_blocks = 40

        # Register the patch for transformer blocks
        for i in range(num_blocks):
            model.set_model_patch_replace(
                freelong_block_patch,
                "dit",
                "double_block",
                i
            )

        actual_end = blend_end_block if blend_end_block != -1 else num_blocks
        blocks_to_patch = actual_end - blend_start_block
        logger.info(f"  Registered patches for {blocks_to_patch} transformer blocks ({blend_start_block} to {actual_end-1})")
        logger.info("=" * 60)
        logger.info("[FreeLong] Patching complete. FreeLong will activate during sampling.")
        logger.info("=" * 60)

        return (model,)


# ============================================================================
# FreeLong Enforcer - Stricter Motion Locking
# ============================================================================

class WanFreeLongEnforcer:
    """
    FreeLong Enforcer: Stricter motion consistency for Wan 2.2 video generation.

    An enhanced version of FreeLong with stronger motion locking:
    1. Early blocks use global-only attention (faster, locks motion early)
    2. 3-tier frequency blending: locked ultra-low, blended mid, local high
    3. Motion skeleton is protected from local attention override

    Use this when standard FreeLong still shows motion drift or direction reversals.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Turn FreeLong Enforcer on or off."
                }),
                "motion_lock_ratio": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "tooltip": "Ultra-low frequencies that are 100% locked to global. Higher = stronger motion lock but may reduce dynamism. 0.15 is a good starting point."
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "How strongly the enforcer affects the video. Higher = more consistent but slightly softer."
                }),
                "low_freq_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Upper bound for global influence (above motion_lock_ratio, frequencies are blended)."
                }),
                "local_window_frames": ("INT", {
                    "default": 33,
                    "min": 9,
                    "max": 241,
                    "step": 4,
                    "tooltip": "Window size for local attention. Smaller = sharper details, larger = smoother."
                }),
                "motion_lock_blocks": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "First N blocks use global-only (no local processing). Establishes motion early. 5 is a good default. 0 disables this feature."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "video/wan"
    DESCRIPTION = "Stricter FreeLong with locked motion skeleton - use when standard FreeLong shows drift"

    def patch_model(
        self,
        model,
        enabled: bool,
        motion_lock_ratio: float,
        blend_strength: float,
        low_freq_ratio: float,
        local_window_frames: int,
        motion_lock_blocks: int
    ):
        """
        Patch the model with enforced motion locking.
        """
        local_window_size = ((local_window_frames - 1) // 4) + 1

        if not enabled:
            logger.info("[FreeLong Enforcer] DISABLED - Model returned without patching")
            return (model,)

        logger.info("=" * 60)
        logger.info("[FreeLong Enforcer] APPLYING ENFORCED MOTION LOCKING")
        logger.info("=" * 60)
        logger.info(f"  Motion Lock Ratio: {motion_lock_ratio:.0%} (ultra-low frequencies locked)")
        logger.info(f"  Blend Ratio: {motion_lock_ratio:.0%} - {low_freq_ratio:.0%} (mid frequencies blended)")
        logger.info(f"  Local Ratio: {low_freq_ratio:.0%} - 100% (high frequencies from local)")
        logger.info(f"  Motion Lock Blocks: first {motion_lock_blocks} blocks (global-only)")
        logger.info(f"  Blend Strength: {blend_strength:.2f}")
        logger.info(f"  Local Window: {local_window_frames} video frames → {local_window_size} latent frames")

        model = model.clone()

        enforcer_settings = {
            "motion_lock_ratio": motion_lock_ratio,
            "low_freq_ratio": low_freq_ratio,
            "blend_strength": blend_strength,
            "local_window_size": local_window_size,
            "motion_lock_blocks": motion_lock_blocks,
            "first_run": True,
        }

        def enforcer_block_patch(block_args, extra_args):
            """
            Enforcer patch with early-block global-only and 3-tier spectral blend.
            """
            original_block = extra_args["original_block"]
            transformer_options = block_args.get("transformer_options", {})
            block_index = transformer_options.get("block_index", 0)

            strength = enforcer_settings["blend_strength"]
            if strength == 0.0:
                return original_block(block_args)

            try:
                x = block_args["img"]
                batch_size, seq_len, hidden_dim = x.shape

                window_size = enforcer_settings["local_window_size"]
                lock_ratio = enforcer_settings["motion_lock_ratio"]
                low_ratio = enforcer_settings["low_freq_ratio"]
                lock_blocks = enforcer_settings["motion_lock_blocks"]

                if enforcer_settings["first_run"]:
                    enforcer_settings["first_run"] = False
                    logger.info("")
                    logger.info("[FreeLong Enforcer] PROCESSING STARTING")
                    logger.info(f"  Input shape: batch={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")

                # ============ GLOBAL STREAM ============
                global_out = original_block(block_args)
                global_x = global_out["img"]

                # Early blocks: global-only (faster, establishes motion)
                if block_index < lock_blocks:
                    if enforcer_settings.get("log_lock_blocks", True) and block_index == 0:
                        enforcer_settings["log_lock_blocks"] = False
                        logger.info(f"  Blocks 0-{lock_blocks-1}: Global-only (motion locking phase)")
                    return {"img": global_x}

                # ============ LOCAL STREAM (same as FreeLong) ============
                cached_seq_len = enforcer_settings.get("cached_seq_len")
                if cached_seq_len == seq_len:
                    num_frames = enforcer_settings["cached_num_frames"]
                    spatial_size = enforcer_settings["cached_spatial_size"]
                    freqs_seq_dim = enforcer_settings.get("cached_freqs_seq_dim")
                    overlap = enforcer_settings["cached_overlap"]
                    stride = enforcer_settings["cached_stride"]
                else:
                    num_frames = None
                    for candidate_frames in [21, 17, 13, 9, 5]:
                        if seq_len % candidate_frames == 0:
                            num_frames = candidate_frames
                            break

                    if num_frames is None or num_frames <= window_size:
                        # Fallback: apply enforced blend to global only
                        blended_x = freelong_enforced_blend(global_x, global_x, lock_ratio, low_ratio, strength)
                        return {"img": blended_x}

                    spatial_size = seq_len // num_frames
                    overlap = window_size // 2
                    stride = window_size - overlap

                    freqs = block_args.get("pe")
                    freqs_seq_dim = None
                    if freqs is not None:
                        for dim_idx, dim_size in enumerate(freqs.shape):
                            if dim_size == seq_len:
                                freqs_seq_dim = dim_idx
                                break

                    enforcer_settings["cached_seq_len"] = seq_len
                    enforcer_settings["cached_num_frames"] = num_frames
                    enforcer_settings["cached_spatial_size"] = spatial_size
                    enforcer_settings["cached_freqs_seq_dim"] = freqs_seq_dim
                    enforcer_settings["cached_overlap"] = overlap
                    enforcer_settings["cached_stride"] = stride

                    if overlap > 0:
                        ramp_up = torch.linspace(0, 1, overlap + 1, device=x.device, dtype=x.dtype)[1:]
                        ramp_down = torch.linspace(1, 0, overlap + 1, device=x.device, dtype=x.dtype)[:-1]
                        enforcer_settings["cached_ramp_up"] = ramp_up
                        enforcer_settings["cached_ramp_down"] = ramp_down

                    if enforcer_settings.get("log_structure", True):
                        enforcer_settings["log_structure"] = False
                        logger.info(f"  Blocks {lock_blocks}+: Dual-stream with enforced 3-tier blend")
                        logger.info(f"  Detected: {num_frames} frames × {spatial_size} spatial tokens")

                freqs = block_args.get("pe")
                x_temporal = x.view(batch_size, num_frames, spatial_size, hidden_dim)

                local_x_temporal = torch.zeros(batch_size, num_frames, spatial_size, hidden_dim,
                                               device=x.device, dtype=x.dtype)
                weight_accumulator = torch.zeros(batch_size, num_frames, 1, 1,
                                                  device=x.device, dtype=x.dtype)

                cached_ramp_up = enforcer_settings.get("cached_ramp_up")
                cached_ramp_down = enforcer_settings.get("cached_ramp_down")

                original_img = block_args.get("img")
                original_pe = block_args.get("pe")

                start_frame = 0
                while start_frame < num_frames:
                    end_frame = min(start_frame + window_size, num_frames)
                    window_frames = end_frame - start_frame

                    window_x = x_temporal[:, start_frame:end_frame, :, :]
                    window_flat = window_x.reshape(batch_size, window_frames * spatial_size, hidden_dim)

                    block_args["img"] = window_flat

                    if freqs is not None and freqs_seq_dim is not None:
                        start_token = start_frame * spatial_size
                        end_token = end_frame * spatial_size
                        slices = [slice(None)] * len(freqs.shape)
                        slices[freqs_seq_dim] = slice(start_token, end_token)
                        block_args["pe"] = freqs[tuple(slices)]

                    window_out = original_block(block_args)
                    window_result = window_out["img"]
                    window_result = window_result.view(batch_size, window_frames, spatial_size, hidden_dim)

                    weights = torch.ones(window_frames, device=x.device, dtype=x.dtype)

                    if start_frame > 0 and overlap > 0 and cached_ramp_up is not None:
                        ramp_len = min(overlap, window_frames)
                        weights[:ramp_len] = cached_ramp_up[:ramp_len]

                    if end_frame < num_frames and overlap > 0 and cached_ramp_down is not None:
                        ramp_len = min(overlap, window_frames)
                        weights[-ramp_len:] = weights[-ramp_len:] * cached_ramp_down[:ramp_len]

                    weights = weights.view(1, window_frames, 1, 1)

                    weighted_result = window_result * weights
                    local_x_temporal[:, start_frame:end_frame].add_(weighted_result)
                    weight_accumulator[:, start_frame:end_frame].add_(weights)

                    del window_x, window_flat, window_out
                    del window_result, weights, weighted_result

                    start_frame += stride
                    if start_frame >= num_frames:
                        break

                block_args["img"] = original_img
                if original_pe is not None:
                    block_args["pe"] = original_pe

                weight_accumulator = weight_accumulator.clamp(min=1e-8)
                local_x_temporal.div_(weight_accumulator)
                del weight_accumulator
                local_x = local_x_temporal.reshape(batch_size, seq_len, hidden_dim)
                del local_x_temporal, x_temporal

                # ============ ENFORCED 3-TIER BLEND ============
                blended_x = freelong_enforced_blend(global_x, local_x, lock_ratio, low_ratio, strength)

                del global_x, local_x

                return {"img": blended_x}

            except Exception as e:
                logger.error(f"[FreeLong Enforcer] Error in block {block_index}: {e}", exc_info=True)
                return original_block(block_args)

        num_blocks = 40
        for i in range(num_blocks):
            model.set_model_patch_replace(
                enforcer_block_patch,
                "dit",
                "double_block",
                i
            )

        logger.info(f"  Registered enforcer patches for {num_blocks} transformer blocks")
        logger.info("=" * 60)
        logger.info("[FreeLong Enforcer] Patching complete.")
        logger.info("=" * 60)

        return (model,)


# ============================================================================
# WanMotionScale - Temporal RoPE Scaling
# ============================================================================

class WanMotionScale:
    """
    Scale temporal position embeddings to control motion speed in Wan video generation.

    Uses ComfyUI's built-in rope_options to scale temporal positions at the source,
    making the model "think" frames are further apart in time.

    Use case: Generate with scale_t > 1, then RIFE interpolate to fill gaps,
    effectively getting longer videos with more motion coverage.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable/disable motion scaling."
                }),
                "scale_t": ("FLOAT", {
                    "default": 1.5,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Temporal scale. >1 = faster motion, <1 = slower, <0 = reversed time positions. Experiment freely!"
                }),
            },
            "optional": {
                "scale_y": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Height/Y scale. Affects vertical spatial encoding."
                }),
                "scale_x": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Width/X scale. Affects horizontal spatial encoding."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "video/wan"
    DESCRIPTION = "Scale temporal positions to control motion speed - use with RIFE for longer videos"

    def patch_model(
        self,
        model,
        enabled: bool,
        scale_t: float,
        scale_y: float = 1.0,
        scale_x: float = 1.0,
    ):
        """
        Patch the model to scale RoPE positions via transformer_options.
        """
        if not enabled:
            logger.info("[WanMotionScale] DISABLED - Model returned without patching")
            return (model,)

        if scale_t == 1.0 and scale_y == 1.0 and scale_x == 1.0:
            logger.info("[WanMotionScale] All scales are 1.0 - Model returned without patching")
            return (model,)

        logger.info("=" * 60)
        logger.info("[WanMotionScale] TEMPORAL ROPE SCALING")
        logger.info("=" * 60)
        logger.info(f"  Temporal Scale (scale_t): {scale_t:.2f}x")
        if scale_y != 1.0:
            logger.info(f"  Height Scale (scale_y): {scale_y:.2f}x")
        if scale_x != 1.0:
            logger.info(f"  Width Scale (scale_x): {scale_x:.2f}x")
        logger.info("=" * 60)

        model = model.clone()

        # Set rope_options in model_options which gets passed to transformer_options
        rope_options = {
            "scale_t": scale_t,
            "scale_y": scale_y,
            "scale_x": scale_x,
        }

        # Add rope_options to the model's transformer options
        model.model_options.setdefault("transformer_options", {})["rope_options"] = rope_options

        logger.info("[WanMotionScale] rope_options added to transformer_options")
        logger.info("=" * 60)

        return (model,)
