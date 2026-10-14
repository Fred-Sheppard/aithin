"""Shared image preprocessing for Ogham model inputs."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image, target_size=(128, 128), contrast_cutoff=1.0):
    """Convert an image to normalized grayscale model input.

    Autocontrast suppresses small background and paper-tone variations while
    preserving the character geometry. Resizing remains nearest-neighbour to
    match the model's existing synthetic training inputs.

    Args:
        image: A PIL image or a path to an image file.
        target_size: (height, width) expected by the model.
        contrast_cutoff: Percent of histogram extremes ignored by autocontrast.

    Returns:
        Float32 array shaped (height, width, 1), with values in [0, 1].
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            grayscale = opened.convert("L")
    else:
        grayscale = image.convert("L")

    grayscale = ImageOps.autocontrast(grayscale, cutoff=contrast_cutoff)
    height, width = target_size
    grayscale = grayscale.resize((width, height), resample=Image.Resampling.NEAREST)

    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    return array[..., np.newaxis]
