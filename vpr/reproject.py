"""
Convert equirectangular panoramas to 4 perspective views (N/E/S/W).
Uses the equilib library for equirectangular-to-perspective projection.
"""
import numpy as np
from PIL import Image
from tqdm import tqdm
import equilib

from . import config

DIRECTIONS = [
    ("N", 0.0),
    ("E", 90.0),
    ("S", 180.0),
    ("W", 270.0),
]


def reproject_panorama(pano_path, output_dir, pano_id):
    img = np.array(Image.open(pano_path).convert("RGB"))
    # equilib expects (C, H, W) for numpy
    img_chw = np.transpose(img, (2, 0, 1)).astype(np.float32)

    results = []
    for direction, yaw_deg in DIRECTIONS:
        rots = {"roll": 0.0, "pitch": 0.0, "yaw": np.radians(yaw_deg)}
        persp = equilib.equi2pers(
            equi=img_chw,
            rots=rots,
            height=config.PERSPECTIVE_SIZE,
            width=config.PERSPECTIVE_SIZE,
            fov_x=float(config.PERSPECTIVE_FOV),
            mode="bilinear",
        )
        # Convert back to (H, W, C) uint8
        persp_hwc = np.transpose(persp, (1, 2, 0)).astype(np.uint8)
        out_path = output_dir / f"{pano_id}_{direction}.jpg"
        Image.fromarray(persp_hwc).save(out_path, quality=90)
        results.append(out_path)

    return results


def main():
    config.PERSPECTIVE_DIR.mkdir(parents=True, exist_ok=True)

    pano_files = sorted(config.PANORAMA_DIR.glob("*.jpg"))
    if not pano_files:
        print(f"No panoramas found in {config.PANORAMA_DIR}")
        return

    print(f"Reprojecting {len(pano_files)} panoramas to perspective views...")
    total_output = 0
    for pano_path in tqdm(pano_files, desc="Reprojecting"):
        pano_id = pano_path.stem
        first_persp = config.PERSPECTIVE_DIR / f"{pano_id}_N.jpg"
        if first_persp.exists():
            continue
        reproject_panorama(pano_path, config.PERSPECTIVE_DIR, pano_id)
        total_output += len(DIRECTIONS)

    print(f"Done: {total_output} perspective images saved to {config.PERSPECTIVE_DIR}")


if __name__ == "__main__":
    main()
