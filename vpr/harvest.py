"""
Harvest Google Street View panoramas for the target bounding box using streetlevel.
Tile-based processing: divides the area into ~500m tiles, downloads panoramas,
reprojects to perspective views, then deletes panoramas to save disk space.
Supports resuming — skips already-processed panoramas and perspectives.
"""
import asyncio
import json
import math
import aiohttp
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from streetlevel import streetview
import equilib

from . import config

TILE_SIZE_M = 500

DIRECTIONS = [
    ("N", 0.0),
    ("E", 90.0),
    ("S", 180.0),
    ("W", 270.0),
]


def generate_tiles():
    lat_step = TILE_SIZE_M / config.METERS_PER_DEGREE_LAT
    lon_step = TILE_SIZE_M / config.METERS_PER_DEGREE_LON
    tiles = []
    lat = config.MIN_LAT
    while lat < config.MAX_LAT:
        lon = config.MIN_LON
        while lon < config.MAX_LON:
            tiles.append((
                lat, lon,
                min(lat + lat_step, config.MAX_LAT),
                min(lon + lon_step, config.MAX_LON),
            ))
            lon += lon_step
        lat += lat_step
    return tiles


def generate_grid_for_tile(min_lat, min_lon, max_lat, max_lon):
    points = []
    lat = min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            points.append((lat, lon))
            lon += config.LON_STEP
        lat += config.LAT_STEP
    return points


async def find_panoramas(points, existing_pano_ids):
    found = {}
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

    async def find_one(session, lat, lon):
        async with semaphore:
            try:
                pano = await streetview.find_panorama_async(
                    lat, lon, session, radius=config.FIND_RADIUS_M
                )
                if pano and not pano.is_third_party and pano.id not in found and pano.id not in existing_pano_ids:
                    found[pano.id] = pano
            except Exception as e:
                pass

    async with aiohttp.ClientSession() as session:
        tasks = [find_one(session, lat, lon) for lat, lon in points]
        for coro in asyncio.as_completed(tasks):
            await coro

    return found


async def download_panoramas(panoramas):
    config.PANORAMA_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    failed = []

    async def download_one(session, pano_id, pano):
        path = config.PANORAMA_DIR / f"{pano_id}.jpg"
        if path.exists():
            return
        async with semaphore:
            try:
                await streetview.download_panorama_async(
                    pano, str(path), session, zoom=config.PANORAMA_ZOOM
                )
            except Exception as e:
                failed.append(pano_id)

    async with aiohttp.ClientSession() as session:
        tasks = [download_one(session, pid, p) for pid, p in panoramas.items()]
        for coro in asyncio.as_completed(tasks):
            await coro

    return failed


def reproject_panorama(pano_path, output_dir, pano_id):
    first_persp = output_dir / f"{pano_id}_N.jpg"
    if first_persp.exists():
        return 0

    img = np.array(Image.open(pano_path).convert("RGB"))
    img_chw = np.transpose(img, (2, 0, 1)).astype(np.float32)

    count = 0
    for direction, yaw_deg in DIRECTIONS:
        out_path = output_dir / f"{pano_id}_{direction}.jpg"
        if out_path.exists():
            continue
        rots = {"roll": 0.0, "pitch": 0.0, "yaw": np.radians(yaw_deg)}
        persp = equilib.equi2pers(
            equi=img_chw,
            rots=rots,
            height=config.PERSPECTIVE_SIZE,
            width=config.PERSPECTIVE_SIZE,
            fov_x=float(config.PERSPECTIVE_FOV),
            mode="bilinear",
        )
        persp_hwc = np.transpose(persp, (1, 2, 0)).astype(np.uint8)
        Image.fromarray(persp_hwc).save(out_path, quality=90)
        count += 1
    return count


def load_metadata():
    if config.METADATA_FILE.exists():
        with open(config.METADATA_FILE) as f:
            return json.load(f)
    return {}


def save_metadata(metadata):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


COMPLETED_TILES_FILE = config.DATA_DIR / "completed_tiles.json"


def load_completed_tiles():
    if COMPLETED_TILES_FILE.exists():
        with open(COMPLETED_TILES_FILE) as f:
            return set(json.load(f))
    return set()


def save_completed_tiles(completed):
    with open(COMPLETED_TILES_FILE, "w") as f:
        json.dump(sorted(completed), f)


def get_already_processed_pano_ids():
    if not config.PERSPECTIVE_DIR.exists():
        return set()
    return {p.stem.rsplit("_", 1)[0] for p in config.PERSPECTIVE_DIR.glob("*_N.jpg")}


async def main():
    config.PANORAMA_DIR.mkdir(parents=True, exist_ok=True)
    config.PERSPECTIVE_DIR.mkdir(parents=True, exist_ok=True)

    tiles = generate_tiles()
    print(f"Target area: {len(tiles)} tiles ({TILE_SIZE_M}m x {TILE_SIZE_M}m)")

    metadata = load_metadata()
    already_processed = get_already_processed_pano_ids()
    completed_tiles = load_completed_tiles()
    print(f"Already processed: {len(already_processed)} panoramas, {len(completed_tiles)} tiles complete")

    total_new_panos = 0
    total_new_persps = 0

    for i, (t_min_lat, t_min_lon, t_max_lat, t_max_lon) in enumerate(tiles):
        if i in completed_tiles:
            continue

        points = generate_grid_for_tile(t_min_lat, t_min_lon, t_max_lat, t_max_lon)
        panos = await find_panoramas(points, already_processed)

        if not panos:
            completed_tiles.add(i)
            continue

        print(f"Tile {i+1}/{len(tiles)}: {len(panos)} new panoramas from {len(points)} grid points")

        for pid, pano in panos.items():
            date_str = str(pano.date) if pano.date else None
            metadata[pid] = {
                "lat": pano.lat,
                "lon": pano.lon,
                "heading": pano.heading,
                "date": date_str,
            }

        failed = await download_panoramas(panos)

        tile_persps = 0
        for pid in panos:
            if pid in failed:
                continue
            pano_path = config.PANORAMA_DIR / f"{pid}.jpg"
            if not pano_path.exists():
                continue
            tile_persps += reproject_panorama(pano_path, config.PERSPECTIVE_DIR, pid)
            pano_path.unlink()
            already_processed.add(pid)

        total_new_panos += len(panos) - len(failed)
        total_new_persps += tile_persps
        completed_tiles.add(i)
        save_completed_tiles(completed_tiles)

        if (i + 1) % 10 == 0:
            save_metadata(metadata)
            print(f"  Progress: {total_new_panos} panos, {total_new_persps} perspectives, "
                  f"tile {i+1}/{len(tiles)}")

    save_metadata(metadata)

    total_persps = len(list(config.PERSPECTIVE_DIR.glob("*.jpg")))
    total_panos_in_meta = len(metadata)
    print(f"\nDone: {total_panos_in_meta} total panoramas, {total_persps} total perspectives")
    print(f"  This run: {total_new_panos} new panoramas, {total_new_persps} new perspectives")


if __name__ == "__main__":
    asyncio.run(main())
