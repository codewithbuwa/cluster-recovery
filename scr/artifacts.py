from pathlib import Path
from shutil import copy2


ROOT = Path(__file__).resolve().parents[1]


def copy_to_latex_images(path: Path) -> Path:
    return copy_to_image_dir(path, ROOT / "latex" / "images")


def copy_to_beamer_images(path: Path) -> Path:
    return copy_to_image_dir(path, ROOT / "beamer" / "images")


def copy_to_latex_and_beamer_images(path: Path) -> list[Path]:
    return [
        copy_to_latex_images(path),
        copy_to_beamer_images(path),
    ]


def copy_to_image_dir(path: Path, image_dir: Path) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    target = image_dir / path.name
    copy2(path, target)
    return target


def copy_to_mixed_images(path: Path) -> list[Path]:
    return [
        copy_to_image_dir(path, ROOT / "mixed_report" / "images"),
        copy_to_image_dir(path, ROOT / "mixed_beamer" / "images"),
        copy_to_image_dir(path, ROOT / "beamer" / "images"),
    ]
