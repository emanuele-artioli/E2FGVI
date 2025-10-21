"""Command line entry point for the E2FGVI package."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import gdown

from . import __version__  # noqa: F401  # expose version on CLI import
from .test import build_parser, run_inference

_DRIVE_FILE_ID = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"
_DEFAULT_CKPT_NAME = "E2FGVI-HQ-CVPR22.pth"


def _download_checkpoint(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading E2FGVI-HQ checkpoint to {destination}...")
    gdown.download(id=_DRIVE_FILE_ID, output=str(destination), quiet=False)


def _resolve_checkpoint_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    package_root = Path(__file__).resolve().parent
    package_candidate = (package_root / candidate).resolve()
    if package_candidate.exists():
        return package_candidate

    release_dir = package_root / "release_model"
    target = release_dir / _DEFAULT_CKPT_NAME
    if not target.exists():
        _download_checkpoint(target)
    return target


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    ckpt_path = _resolve_checkpoint_path(args.ckpt)
    if not ckpt_path.exists():
        target = ckpt_path if ckpt_path.name.endswith(".pth") else ckpt_path / _DEFAULT_CKPT_NAME
        if not target.exists():
            _download_checkpoint(target)
        ckpt_path = target

    args.ckpt = str(ckpt_path)
    run_inference(args)


if __name__ == "__main__":
    main(sys.argv[1:])