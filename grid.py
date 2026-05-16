"""Render a labeled sample grid: class names down the left, one row per class."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid


def save_labeled_grid(imgs, class_names, path, nrow=10, scale=4, pad=2):
    """Save a grid with a class label beside each row.

    imgs: (len(class_names) * nrow, C, H, W) tensor in [0, 1], ordered so that
    the first `nrow` images are class 0, the next `nrow` class 1, and so on.
    """
    grid = make_grid(imgs.clamp(0, 1), nrow=nrow, padding=pad)
    arr = (grid * 255).byte().permute(1, 2, 0).cpu().numpy()
    if arr.shape[2] == 1:                      # grayscale -> RGB
        arr = np.repeat(arr, 3, axis=2)
    img = Image.fromarray(arr)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    margin = 150
    canvas = Image.new("RGB", (img.width + margin, img.height), "white")
    canvas.paste(img, (margin, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    row_h = img.height / len(class_names)
    for i, name in enumerate(class_names):
        draw.text((margin - 14, row_h * (i + 0.5)), name,
                  fill="black", anchor="rm", font=font)
    canvas.save(path)
