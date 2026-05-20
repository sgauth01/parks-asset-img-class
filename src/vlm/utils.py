import math
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent 

def show_asset_images(df, asset_id, max_cols=4, figsize=(6, 6)):
    """
    Display all images for a given asset_id in a tiled grid.
    """

    # Get paths
    paths = (
        df.loc[df['asset_id'] == asset_id, "image_path"]
        .dropna()
        .tolist()
    )

    if len(paths) == 0:
        print("No images found for this asset.")
        return

    # Fix paths if necessary
    # (Modify this if your image directory differs)
    fixed_paths = [str(ROOT / p.replace("data/", "data/raw/", 1)) for p in paths]

    # Grid layout
    n = len(fixed_paths)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes for easier indexing
    axes = axes.flatten() if n > 1 else [axes]

    for ax, img_path in zip(axes, fixed_paths):
        if not os.path.exists(img_path):
            ax.set_title("Missing")
            ax.axis("off")
            continue

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # correct orientation

        ax.imshow(img)
        ax.axis("off")

    # Turn off any extra axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()