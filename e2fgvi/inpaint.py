# -*- coding: utf-8 -*-
"""Functional API for E2FGVI video inpainting with numpy array I/O."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Literal

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import gdown

from .core.utils import to_tensors

# Checkpoint download settings
_DRIVE_FILE_ID = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"
_DEFAULT_CKPT_NAME = "E2FGVI-HQ-CVPR22.pth"
_PACKAGE_ROOT = Path(__file__).resolve().parent
_RELEASE_DIR = _PACKAGE_ROOT / "release_model"


def _download_checkpoint(destination: Path) -> None:
    """Download E2FGVI-HQ checkpoint from Google Drive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading E2FGVI-HQ checkpoint to {destination}...")
    gdown.download(id=_DRIVE_FILE_ID, output=str(destination), quiet=False)


def _ensure_checkpoint() -> Path:
    """Ensure the checkpoint exists, downloading if necessary."""
    target = _RELEASE_DIR / _DEFAULT_CKPT_NAME
    if not target.exists():
        _download_checkpoint(target)
    return target


@dataclass
class InpaintingConfig:
    """Configuration for E2FGVI video inpainting.
    
    Attributes:
        model: Model variant to use ('e2fgvi' or 'e2fgvi_hq').
        checkpoint_path: Path to model checkpoint. If None, downloads E2FGVI-HQ.
        step: Stride of global reference frames (default: 10).
        num_ref: Number of reference frames (-1 for all, default: -1).
        neighbor_stride: Local neighbor stride (default: 5).
        mask_dilation: Mask dilation iterations (default: 4).
        width: Target width (None = use original, must be set for e2fgvi).
        height: Target height (None = use original, must be set for e2fgvi).
        device: Device to run inference on (default: auto-detect).
    """
    model: Literal["e2fgvi", "e2fgvi_hq"] = "e2fgvi_hq"
    checkpoint_path: Optional[str] = None
    step: int = 10
    num_ref: int = -1
    neighbor_stride: int = 5
    mask_dilation: int = 4
    width: Optional[int] = None
    height: Optional[int] = None
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # e2fgvi (non-HQ) requires fixed size
        if self.model == "e2fgvi":
            if self.width is None:
                self.width = 432
            if self.height is None:
                self.height = 240


class E2FGVIModel:
    """E2FGVI model wrapper for video inpainting.
    
    This class loads and caches the model for efficient repeated inference.
    
    Example:
        >>> model = E2FGVIModel()
        >>> frames = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)
        >>> masks = np.zeros((10, 480, 640), dtype=np.uint8)
        >>> masks[:, 100:200, 100:200] = 255  # Region to inpaint
        >>> result = model.inpaint(frames, masks)
    """
    
    def __init__(
        self,
        model: Literal["e2fgvi", "e2fgvi_hq"] = "e2fgvi_hq",
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize E2FGVI model.
        
        Args:
            model: Model variant ('e2fgvi' or 'e2fgvi_hq').
            checkpoint_path: Path to checkpoint. If None, downloads E2FGVI-HQ.
            device: Device to run inference on (default: auto-detect).
        """
        self.model_name = model
        self.checkpoint_path = checkpoint_path
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = None
        
    def _load_model(self):
        """Lazy load model on first use."""
        if self._model is not None:
            return
            
        # Resolve checkpoint path
        if self.checkpoint_path is None:
            ckpt_path = _ensure_checkpoint()
        else:
            ckpt_path = Path(self.checkpoint_path)
            if not ckpt_path.exists():
                ckpt_path = _ensure_checkpoint()
                
        # Load model
        module = importlib.import_module(f"e2fgvi.model.{self.model_name}")
        self._model = module.InpaintGenerator().to(self.device)
        checkpoint = torch.load(str(ckpt_path), map_location=self.device)
        self._model.load_state_dict(checkpoint)
        print(f"Loading model from: {ckpt_path}")
        self._model.eval()
        
    def inpaint(
        self,
        frames: np.ndarray,
        masks: np.ndarray,
        config: Optional[InpaintingConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> np.ndarray:
        """Inpaint video frames using E2FGVI.
        
        Args:
            frames: Input video frames as numpy array of shape (T, H, W, 3) with dtype uint8 (RGB).
            masks: Masks indicating regions to inpaint, shape (T, H, W) or (T, H, W, 1).
                   Values > 0 indicate regions to inpaint. Can be uint8 or bool.
            config: Inpainting configuration. If None, uses default settings.
            progress_callback: Optional callback function(current, total) for progress updates.
            
        Returns:
            Inpainted frames as numpy array of shape (T, H, W, 3) with dtype uint8 (RGB).
        """
        if config is None:
            config = InpaintingConfig(
                model=self.model_name,
                checkpoint_path=self.checkpoint_path,
                device=self.device
            )
            
        self._load_model()
        
        return _inpaint_numpy(
            frames=frames,
            masks=masks,
            model=self._model,
            config=config,
            progress_callback=progress_callback
        )


def _get_ref_index(
    frame_idx: int,
    neighbor_ids: List[int],
    length: int,
    ref_length: int,
    num_ref: int,
) -> List[int]:
    """Get reference frame indices."""
    ref_index: List[int] = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, frame_idx - ref_length * (num_ref // 2))
        end_idx = min(length - 1, frame_idx + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) >= num_ref:
                    break
                ref_index.append(i)
    return ref_index


def _prepare_frames_from_numpy(
    frames: np.ndarray,
    size: Optional[Tuple[int, int]] = None
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """Convert numpy frames to PIL Images and resize if needed.
    
    Args:
        frames: Input frames as numpy array of shape (T, H, W, 3) with dtype uint8.
        size: Optional target size as (width, height).
        
    Returns:
        Tuple of (list of PIL Images, final_size).
    """
    pil_frames = [Image.fromarray(f) for f in frames]
    if size is not None:
        pil_frames = [f.resize(size) for f in pil_frames]
        return pil_frames, size
    return pil_frames, pil_frames[0].size


def _prepare_masks_from_numpy(
    masks: np.ndarray,
    size: Tuple[int, int],
    dilation: int = 4
) -> List[Image.Image]:
    """Process numpy masks for inpainting.
    
    Args:
        masks: Masks as numpy array of shape (T, H, W) or (T, H, W, 1).
        size: Target size as (width, height).
        dilation: Dilation iterations.
        
    Returns:
        List of PIL Images with dilated binary masks.
    """
    # Ensure masks are 2D per frame
    if masks.ndim == 4:
        masks = masks.squeeze(-1)
    
    # Normalize to 0-255 uint8
    if masks.dtype == bool:
        masks = masks.astype(np.uint8) * 255
    elif masks.max() <= 1:
        masks = (masks * 255).astype(np.uint8)
    
    processed_masks = []
    for i in range(masks.shape[0]):
        mask_img = Image.open_if_path_else_use(masks[i]) if isinstance(masks[i], (str, Path)) else Image.fromarray(masks[i])
        mask_img = mask_img.resize(size, Image.NEAREST)
        mask_arr = np.array(mask_img.convert("L"))
        binary_mask = np.array(mask_arr > 0).astype(np.uint8)
        if dilation > 0:
            binary_mask = cv2.dilate(
                binary_mask,
                cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                iterations=dilation,
            )
        processed_masks.append(Image.fromarray(binary_mask * 255))
        
    return processed_masks


def _inpaint_numpy(
    frames: np.ndarray,
    masks: np.ndarray,
    model,
    config: InpaintingConfig,
    progress_callback: Optional[callable] = None
) -> np.ndarray:
    """Core inpainting function that works with numpy arrays.
    
    Args:
        frames: Input frames as numpy array of shape (T, H, W, 3).
        masks: Masks as numpy array of shape (T, H, W) or (T, H, W, 1).
        model: E2FGVI model instance.
        config: Inpainting configuration.
        progress_callback: Optional progress callback.
        
    Returns:
        Inpainted frames as numpy array.
    """
    device = config.device
    
    # Determine target size
    original_size = (frames.shape[2], frames.shape[1])  # (W, H)
    if config.width is not None and config.height is not None:
        size = (config.width, config.height)
    else:
        size = original_size
        
    # Prepare frames
    pil_frames, size = _prepare_frames_from_numpy(frames, size if size != original_size else None)
    width, height = size
    video_length = len(pil_frames)
    
    # Convert to tensors
    tensor_frames = to_tensors()(pil_frames).unsqueeze(0) * 2 - 1
    frames_array = [np.array(frame).astype(np.uint8) for frame in pil_frames]
    
    # Prepare masks
    # Ensure masks are 2D per frame
    if masks.ndim == 4:
        masks = masks.squeeze(-1)
    
    # Normalize to 0-255 uint8
    if masks.dtype == bool:
        masks_normalized = masks.astype(np.uint8) * 255
    elif masks.max() <= 1 and masks.dtype != np.uint8:
        masks_normalized = (masks * 255).astype(np.uint8)
    else:
        masks_normalized = masks.astype(np.uint8)
    
    processed_masks = []
    for i in range(masks_normalized.shape[0]):
        mask_img = Image.fromarray(masks_normalized[i])
        mask_img = mask_img.resize(size, Image.NEAREST)
        mask_arr = np.array(mask_img.convert("L"))
        binary_mask = np.array(mask_arr > 0).astype(np.uint8)
        if config.mask_dilation > 0:
            binary_mask = cv2.dilate(
                binary_mask,
                cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                iterations=config.mask_dilation,
            )
        processed_masks.append(Image.fromarray(binary_mask * 255))
    
    binary_masks = [
        np.expand_dims((np.array(mask) != 0).astype(np.uint8), 2) for mask in processed_masks
    ]
    tensor_masks = to_tensors()(processed_masks).unsqueeze(0)
    
    imgs = tensor_frames.to(device)
    tensor_masks = tensor_masks.to(device)
    
    comp_frames: List[np.ndarray | None] = [None] * video_length
    
    ref_length = config.step
    num_ref = config.num_ref
    neighbor_stride = config.neighbor_stride
    
    # Calculate total steps for progress
    total_steps = (video_length + neighbor_stride - 1) // neighbor_stride
    current_step = 0
    
    for frame_idx in tqdm(range(0, video_length, neighbor_stride), desc="E2FGVI inpainting"):
        neighbor_ids = [
            idx
            for idx in range(
                max(0, frame_idx - neighbor_stride),
                min(video_length, frame_idx + neighbor_stride + 1),
            )
        ]
        ref_ids = _get_ref_index(frame_idx, neighbor_ids, video_length, ref_length, num_ref)
        selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
        selected_masks = tensor_masks[:1, neighbor_ids + ref_ids, :, :, :]
        
        with torch.no_grad():
            masked_imgs = selected_imgs * (1 - selected_masks)
            mod_size_h = 60
            mod_size_w = 108
            h_pad = (mod_size_h - height % mod_size_h) % mod_size_h
            w_pad = (mod_size_w - width % mod_size_w) % mod_size_w
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [3])],
                3,
            )[:, :, :, : height + h_pad, :]
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [4])],
                4,
            )[:, :, :, :, : width + w_pad]
            pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
            pred_imgs = pred_imgs[:, :, :height, :width]
            pred_imgs = (pred_imgs + 1) / 2
            pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
            
            for idx, neighbor in enumerate(neighbor_ids):
                pred = np.array(pred_imgs[idx]).astype(np.uint8)
                combined = pred * binary_masks[neighbor] + frames_array[neighbor] * (
                    1 - binary_masks[neighbor]
                )
                if comp_frames[neighbor] is None:
                    comp_frames[neighbor] = combined
                else:
                    comp_frames[neighbor] = (
                        comp_frames[neighbor].astype(np.float32) * 0.5
                        + combined.astype(np.float32) * 0.5
                    )
                    
        current_step += 1
        if progress_callback is not None:
            progress_callback(current_step, total_steps)
            
    # Convert to uint8
    comp_frames = [f.astype(np.uint8) for f in comp_frames]
    
    # Resize back to original size if needed
    if size != original_size:
        comp_frames = [
            cv2.resize(f, original_size, interpolation=cv2.INTER_CUBIC)
            for f in comp_frames
        ]
        
    # Stack into numpy array
    result = np.stack(comp_frames, axis=0)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


def inpaint(
    frames: np.ndarray,
    masks: np.ndarray,
    config: Optional[InpaintingConfig] = None,
    progress_callback: Optional[callable] = None
) -> np.ndarray:
    """Inpaint video frames using E2FGVI.
    
    This is a convenience function that creates an E2FGVIModel instance
    and runs inpainting. For repeated calls, consider using E2FGVIModel
    directly to avoid reloading models.
    
    Args:
        frames: Input video frames as numpy array of shape (T, H, W, 3) with dtype uint8 (RGB).
        masks: Masks indicating regions to inpaint, shape (T, H, W) or (T, H, W, 1).
               Values > 0 indicate regions to inpaint. Can be uint8 or bool.
        config: Inpainting configuration. If None, uses default settings.
        progress_callback: Optional callback function(current, total) for progress updates.
        
    Returns:
        Inpainted frames as numpy array of shape (T, H, W, 3) with dtype uint8 (RGB).
        
    Example:
        >>> import numpy as np
        >>> from e2fgvi import inpaint, InpaintingConfig
        >>> 
        >>> # Load your video frames (T, H, W, 3) RGB uint8
        >>> frames = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)
        >>> 
        >>> # Create masks (T, H, W) - non-zero values indicate regions to inpaint
        >>> masks = np.zeros((10, 480, 640), dtype=np.uint8)
        >>> masks[:, 100:200, 100:200] = 255  # Region to inpaint
        >>> 
        >>> # Inpaint with default settings (uses E2FGVI-HQ)
        >>> result = inpaint(frames, masks)
        >>> 
        >>> # Or with custom config
        >>> config = InpaintingConfig(model='e2fgvi_hq', mask_dilation=8)
        >>> result = inpaint(frames, masks, config=config)
    """
    if config is None:
        config = InpaintingConfig()
        
    model = E2FGVIModel(
        model=config.model,
        checkpoint_path=config.checkpoint_path,
        device=config.device
    )
    return model.inpaint(frames, masks, config=config, progress_callback=progress_callback)
