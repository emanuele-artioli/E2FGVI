"""E2FGVI video inpainting package."""

from importlib.metadata import PackageNotFoundError, version


def __getattr__(name: str):
    if name == "__version__":
        try:
            return version("e2fgvi")
        except PackageNotFoundError:  # pragma: no cover - fallback for dev installs
            return "0.0.0"
    raise AttributeError(name)


__all__ = ["__version__"]
