import os
import base64

def load_asset_images_base64(asset_id, df):
    """
    Load all images for a given asset and return:
    [{"b64": ..., "mime": ...}, ...]
    """
    paths = (
        df[df["asset_id"] == asset_id]["image_path"]
        .dropna()
        .tolist()
    )

    images = []
    for p in paths:
        fixed = p.replace("data/", "../data/raw/", 1)
        if not os.path.exists(fixed):
            print(f"⚠️ Image not found: {fixed}")
            continue

        mime = "image/png" if fixed.lower().endswith(".png") else "image/jpeg"

        with open(fixed, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        images.append({"b64": b64, "mime": mime})

    return images