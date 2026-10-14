"""
Extract DINOv2-SALAD global descriptors from clean perspective images.

Descriptors are streamed to a float16 memory-mapped .npy file, so RAM use is constant
regardless of how many reference images there are.
"""
import json
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import config
from .model_utils import load_salad_model, get_transform, get_device


class ImageListDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.image_paths[i]).convert("RGB"))


def main():
    device = get_device()
    print(f"Using device: {device}")

    clean_list_path = config.FEATURES_DIR / "clean_images.json"
    if not clean_list_path.exists():
        print(f"Clean image list not found at {clean_list_path}")
        print("Run python -m vpr.filter_images first.")
        return

    with open(clean_list_path) as f:
        image_paths = json.load(f)

    if not image_paths:
        print("No clean images to process.")
        return

    print(f"Loading SALAD model...")
    model = load_salad_model(device)
    loader = DataLoader(
        ImageListDataset(image_paths, get_transform()),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    print(f"Model loaded. Extracting features from {len(image_paths)} images...")

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    descriptors = np.lib.format.open_memmap(
        config.DESCRIPTORS_FILE, mode="w+", dtype=np.float16,
        shape=(len(image_paths), config.SALAD_DIM),
    )

    offset = 0
    for batch in tqdm(loader, desc="Extracting"):
        descs = model(batch.to(device, non_blocking=True)).cpu().numpy()
        descriptors[offset:offset + len(descs)] = descs
        offset += len(descs)

    descriptors.flush()
    print(f"Descriptor matrix shape: {descriptors.shape} ({descriptors.dtype})")
    del descriptors

    list_path = config.FEATURES_DIR / "image_list.pkl"
    with open(list_path, "wb") as f:
        pickle.dump(image_paths, f)

    print(f"Saved descriptors to {config.DESCRIPTORS_FILE}")
    print(f"Saved image list to {list_path}")


if __name__ == "__main__":
    main()
