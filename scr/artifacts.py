from pathlib import Path
from shutil import copy2


def copy_to_latex_images(path: Path) -> Path:
    latex_images_dir = Path(__file__).resolve().parents[1] / "latex" / "images"
    latex_images_dir.mkdir(parents=True, exist_ok=True)
    target = latex_images_dir / path.name
    copy2(path, target)
    return target
