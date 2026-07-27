"""
Extract DINOv2-SALAD global descriptors from clean perspective images.
"""
import json
import pickle

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from . import config
from .model_utils import load_salad_model, get_transform


def extract_batch(model, batch_tensors, device):
    batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        descriptors = model(batch)
    return descriptors.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    clean_list_path = config.FEATURES_DIR / "clean_images.json"
    if not clean_list_path.exists():
        print(f"Clean image list not found at {clean_list_path}")
        print("Run 03_filter.py first.")
        return

    with open(clean_list_path) as f:
        image_paths = json.load(f)

    if not image_paths:
        print("No clean images to process.")
        return

    print(f"Loading SALAD model...")
    model = load_salad_model(device)
    transform = get_transform()
    print(f"Model loaded. Extracting features from {len(image_paths)} images...")

    all_descriptors = []
    batch_tensors = []

    for i, img_path in enumerate(tqdm(image_paths, desc="Extracting")):
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img)
        batch_tensors.append(tensor)

        if len(batch_tensors) == config.BATCH_SIZE or i == len(image_paths) - 1:
            descs = extract_batch(model, batch_tensors, device)
            all_descriptors.append(descs)
            batch_tensors = []

    descriptors = np.vstack(all_descriptors)
    print(f"Descriptor matrix shape: {descriptors.shape}")

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    desc_path = config.FEATURES_DIR / "descriptors.npy"
    np.save(desc_path, descriptors)

    list_path = config.FEATURES_DIR / "image_list.pkl"
    with open(list_path, "wb") as f:
        pickle.dump(image_paths, f)

    print(f"Saved descriptors to {desc_path}")
    print(f"Saved image list to {list_path}")


if __name__ == "__main__":
    main()
