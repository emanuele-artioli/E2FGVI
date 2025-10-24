# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from PIL import Image
from tqdm import tqdm

from .core.utils import to_tensors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E2FGVI")
    parser.add_argument("-v", "--video", type=str, required=True)
    parser.add_argument("-c", "--ckpt", type=str, required=True)
    parser.add_argument("-m", "--mask", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["e2fgvi", "e2fgvi_hq"], required=True)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--num_ref", type=int, default=-1)
    parser.add_argument("--neighbor_stride", type=int, default=5)
    parser.add_argument("--savefps", type=int, default=24)
    parser.add_argument("--set_size", action="store_true", default=False)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    return parser


def get_ref_index(
    frame_idx: int,
    neighbor_ids: Sequence[int],
    length: int,
    ref_length: int,
    num_ref: int,
) -> List[int]:
    ref_index: List[int] = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, frame_idx - ref_length * (num_ref // 2))
        # Clamp end_idx to the last valid index (length - 1) to avoid out-of-range access
        end_idx = min(length - 1, frame_idx + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                # Guard so we don't append more than num_ref references
                if len(ref_index) >= num_ref:
                    break
                ref_index.append(i)
    return ref_index


def read_mask(mpath: Path, size: Sequence[int]) -> List[Image.Image]:
    masks: List[Image.Image] = []
    for mask_name in sorted(os.listdir(mpath)):
        mask_path = mpath / mask_name
        mask_img = Image.open(mask_path)
        mask_img = mask_img.resize(size, Image.NEAREST)
        mask_arr = np.array(mask_img.convert("L"))
        binary_mask = np.array(mask_arr > 0).astype(np.uint8)
        binary_mask = cv2.dilate(
            binary_mask,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
            iterations=4,
        )
        masks.append(Image.fromarray(binary_mask * 255))
    return masks


def read_frames(video_path: Path, as_video: bool) -> List[Image.Image]:
    frames: List[Image.Image] = []
    if as_video:
        vidcap = cv2.VideoCapture(str(video_path))
        success, image = vidcap.read()
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image)
            success, image = vidcap.read()
    else:
        for frame_path in sorted(video_path.iterdir()):
            if not frame_path.is_file():
                continue
            image = cv2.imread(str(frame_path))
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image)
    return frames


def resize_frames(frames: Iterable[Image.Image], size: Sequence[int] | None) -> tuple[List[Image.Image], Sequence[int]]:
    frame_list = list(frames)
    if size is not None:
        resized_frames = [frame.resize(size) for frame in frame_list]
        return resized_frames, size
    size = frame_list[0].size
    return frame_list, size


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "e2fgvi":
        size = (432, 240)
    elif args.set_size:
        size = (args.width, args.height)
    else:
        size = None

    module = importlib.import_module(f"e2fgvi.model.{args.model}")
    model = module.InpaintGenerator().to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loading model from: {args.ckpt}")
    model.eval()

    use_mp4 = args.video.endswith(".mp4")
    print(
        f"Loading videos and masks from: {args.video} | INPUT MP4 format: {use_mp4}"
    )
    frames = read_frames(Path(args.video), use_mp4)
    frames, size = resize_frames(frames, size)
    height, width = size[1], size[0]
    video_length = len(frames)
    tensor_frames = to_tensors()(frames).unsqueeze(0) * 2 - 1
    frames_array = [np.array(frame).astype(np.uint8) for frame in frames]

    masks = read_mask(Path(args.mask), size)
    binary_masks = [
        np.expand_dims((np.array(mask) != 0).astype(np.uint8), 2) for mask in masks
    ]
    tensor_masks = to_tensors()(masks).unsqueeze(0)
    imgs, tensor_masks = tensor_frames.to(device), tensor_masks.to(device)
    comp_frames: List[np.ndarray | None] = [None] * video_length

    ref_length = args.step
    num_ref = args.num_ref
    neighbor_stride = args.neighbor_stride
    default_fps = args.savefps

    print("Start test...")
    for frame_idx in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [
            idx
            for idx in range(
                max(0, frame_idx - neighbor_stride),
                min(video_length, frame_idx + neighbor_stride + 1),
            )
        ]
        ref_ids = get_ref_index(frame_idx, neighbor_ids, video_length, ref_length, num_ref)
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

    print("Saving videos...")
    save_dir = Path(__file__).resolve().parent / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(args.video).name
    save_name = base_name.replace(".mp4", "_results.mp4") if use_mp4 else f"{base_name}_results.mp4"
    save_path = save_dir / save_name
    writer = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        default_fps,
        size,
    )
    for frame in range(video_length):
        comp = comp_frames[frame].astype(np.uint8)  # type: ignore[union-attr]
        writer.write(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    writer.release()
    print(f"Finish test! The result video is saved in: {save_path}.")

    print("Let us enjoy the result!")
    fig = plt.figure("Let us enjoy the result")
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis("off")
    ax1.set_title("Original Video")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.set_title("Our Result")
    imdata1 = ax1.imshow(frames_array[0])
    imdata2 = ax2.imshow(comp_frames[0].astype(np.uint8))

    def update(idx: int) -> None:
        imdata1.set_data(frames_array[idx])
        imdata2.set_data(comp_frames[idx].astype(np.uint8))

    fig.tight_layout()
    animation.FuncAnimation(fig, update, frames=len(frames_array), interval=50)
    plt.show()


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_inference(args)


if __name__ == "__main__":
    main()
