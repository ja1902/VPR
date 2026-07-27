"""
Filter perspective images: remove blurry and too-dark images.
Outputs a clean image list for feature extraction.
"""
import json
import cv2
import numpy as np
from tqdm import tqdm

from . import config


def is_blurry(image_path, threshold=config.BLUR_THRESHOLD):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance < threshold


def is_too_dark(image_path, threshold=config.DARKNESS_THRESHOLD):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    return np.mean(img) < threshold


def main():
    persp_files = sorted(config.PERSPECTIVE_DIR.glob("*.jpg"))
    if not persp_files:
        print(f"No perspective images found in {config.PERSPECTIVE_DIR}")
        return

    print(f"Filtering {len(persp_files)} perspective images...")
    clean = []
    rejected_blur = 0
    rejected_dark = 0

    for img_path in tqdm(persp_files, desc="Filtering"):
        if is_too_dark(img_path):
            rejected_dark += 1
        elif is_blurry(img_path):
            rejected_blur += 1
        else:
            clean.append(str(img_path))

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    clean_list_path = config.FEATURES_DIR / "clean_images.json"
    with open(clean_list_path, "w") as f:
        json.dump(clean, f, indent=2)

    print(f"\nResults:")
    print(f"  Passed:         {len(clean)}")
    print(f"  Rejected blur:  {rejected_blur}")
    print(f"  Rejected dark:  {rejected_dark}")
    print(f"  Clean list saved to {clean_list_path}")


if __name__ == "__main__":
    main()
