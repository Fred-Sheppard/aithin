"""Torch-free Ogham letters, image loading, labelled-sample listing and progress bars, shared by the
ConvNeXt and Keras pipelines."""
import os
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

DIRECTIONS = ["RIGHT", "LEFT", "DIAGONAL", "HORIZONTAL"]
LETTER_TABLE = [
    ["B", "L", "F", "S", "N"],
    ["H", "D", "T", "C", "Q"],
    ["M", "G", "NG", "Z", "R"],
    ["A", "O", "U", "E", "I"],
]
LETTERS = [letter for row in LETTER_TABLE for letter in row]
LETTER_TO_INDEX = {letter: i for i, letter in enumerate(LETTERS)}


def letter_direction(letter):
    return LETTER_TO_INDEX[letter] // 5


def letter_count_index(letter):
    """Zero-based stroke count index (count - 1)."""
    return LETTER_TO_INDEX[letter] % 5


def terminal_width():
    # Output is usually piped through tee, so ask the controlling terminal rather than stderr.
    try:
        with open("/dev/tty") as tty:
            return os.get_terminal_size(tty.fileno()).columns
    except OSError:
        return 100


class StatusBar(tqdm):
    """tqdm bar that also mirrors every active bar into OGHAM_STATUS_FILE, so progress is
    readable with `cat` when there is no terminal to draw on."""
    status_file = os.environ.get("OGHAM_STATUS_FILE")
    lines = {}

    def display(self, msg=None, pos=None):
        super().display(msg, pos)
        if self.status_file:
            StatusBar.lines[self.desc] = self.__str__()
            StatusBar.write_status()

    def close(self):
        super().close()
        if self.status_file and not self.leave:
            StatusBar.lines.pop(self.desc, None)
            StatusBar.write_status()

    @classmethod
    def write_status(cls):
        text = f"updated {time.strftime('%H:%M:%S')}\n" + "".join(line + "\n" for line in cls.lines.values())
        tmp = cls.status_file + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, cls.status_file)


def progress(iterable=None, **kwargs):
    # Without a terminal the bar is only read from logs and the status file, so redraw less often.
    interval = 0.1 if sys.stderr.isatty() else 2.0
    return StatusBar(iterable, ncols=terminal_width() - 1, mininterval=interval, **kwargs)


def load_grayscale(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("L")


def synthetic_samples(output_dir=REPO_ROOT / "output"):
    output_dir = Path(output_dir)
    df = pd.read_csv(output_dir / "labels.csv", dtype=str)
    # Legacy synthesis writes temp-letters-dir/, synthesis_v2 writes images/
    images_dir = output_dir / "images"
    if not images_dir.is_dir():
        images_dir = output_dir / "temp-letters-dir"
    paths = [images_dir / f"{int(n):05d}.png" for n in df["img_no"]]
    return paths, list(df["img_letter"])


def find_image(images_dir, img_no):
    img_no = str(img_no).strip()
    name = f"{int(img_no):05d}" if img_no.isdigit() else img_no
    for p in sorted(Path(images_dir).glob(f"{name}.*")):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            return p
    return None


def csv_samples(csv_file, images_dir):
    """Rows of `img_no,img_letter` resolved to image paths. Missing images and unknown letters are skipped."""
    df = pd.read_csv(csv_file, dtype=str)
    paths, letters = [], []
    for img_no, letter in zip(df["img_no"], df["img_letter"]):
        letter = str(letter).strip()
        if letter not in LETTER_TO_INDEX:
            continue
        path = find_image(images_dir, img_no)
        if path is None:
            print(f"Warning: no image for {img_no} in {images_dir}")
            continue
        paths.append(path)
        letters.append(letter)
    return paths, letters


def support_samples(support_dirs):
    paths, letters = [], []
    for d in support_dirs:
        p, l = csv_samples(Path(d) / "labels.csv", Path(d) / "images")
        paths += p
        letters += l
    return paths, letters
