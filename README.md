# Visual Place Recognition

Visual geolocation system. Given a photo or video, it identifies where it was taken by matching against a pre-built reference database of street-level imagery.

## v0.2.0 — What Changed

### Out-of-range detection

The system now detects when a query is outside the reference area instead of silently returning the nearest (wrong) match.

**Video queries — per-frame geographic clustering:**
- Each video frame is queried against FAISS individually instead of being pooled into a single descriptor
- The per-frame top-1 matches are clustered geographically: if enough frames agree on a location (within 100m), the query is in-range
- If fewer than 40% of frames (or fewer than 3 frames) cluster together, the query is flagged as out-of-range
- This naturally handles shaky or low-quality video — good frames still cluster while junk frames (pointing at the ground, sky, etc.) are filtered out as outliers

**Image queries — Pair-VPR score thresholding:**
- When `--rerank` is used and Pair-VPR runs, the best re-ranking score is checked against a threshold
- A score below 0 indicates no genuine visual match exists and the query is flagged as out-of-range
- Requires `--rerank` to be passed (Pair-VPR must run for the check to work)

### Video pipeline overhaul

- **Removed SeqGeM pooling** — previously, all frame descriptors were pooled into a single query vector. This broke when frames showed diverse scenes (e.g. driving through a city) because the pooled descriptor became too generic to match anything
- **Per-frame FAISS search** — each frame now queries the index independently, producing its own top-k matches
- **Inlier-based result merging** — after geographic clustering identifies reliable frames, their FAISS results are merged (deduplicated by reference image, keeping the best distance) to produce the final ranked list
- **Pair-VPR re-ranking** now uses the first inlier frame as the query image instead of always using frame 0

## How It Works

```
┌─────────────────────────── OFFLINE (setup) ──────────────────────────────┐
│                                                                          │
│  Coordinates ──► Harvest Street  ──► Reproject to  ──► Filter blurry/   │
│  (bounding box)   View panoramas     6 perspectives     dark images      │
│                                                                          │
│                  ──► Extract SALAD  ──► Build FAISS                      │
│                      descriptors       search index                      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── ONLINE (image query) ─────────────────────────┐
│                                                                          │
│  Photo ──► SALAD descriptor ──► FAISS nearest  ──► Pair-VPR             │
│            (8448-dim)           neighbour search    re-ranking*          │
│                                                                          │
│          ──► OOR check** ──► Weighted centroid ──► GPS coordinates       │
│                                                                          │
│  * Re-ranking is adaptive: only triggered when top matches are close    │
│  ** Out-of-range detection via Pair-VPR score (requires --rerank)       │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── ONLINE (video query) ─────────────────────────┐
│                                                                          │
│  Video ──► Extract frames ──► Per-frame SALAD ──► Per-frame FAISS       │
│                                descriptors        search                │
│                                                                          │
│          ──► Geographic clustering ──► OOR check ──► Merge inlier       │
│              of per-frame matches                    results             │
│                                                                          │
│          ──► Pair-VPR re-ranking* ──► Weighted centroid ──► GPS         │
│                                                                          │
│  * Adaptive, uses first inlier frame as query                           │
└──────────────────────────────────────────────────────────────────────────┘
```

**Core components:**

- **DINOv2-SALAD** — Produces 8448-dimensional global descriptors that capture visual identity of a scene, invariant to lighting and viewpoint changes
- **FAISS IndexFlatL2** — Brute-force L2 search over the reference descriptor database for fast nearest-neighbour retrieval
- **Pair-VPR** — Cross-attention re-ranker using DINOv2 ViT-B/14 dense features, triggered adaptively when FAISS results are ambiguous (top-1/top-2 margin < 0.15)
- **Per-frame voting** — Video queries search each frame independently; geographic clustering of per-frame matches determines consensus and filters outliers
- **Weighted centroid** — Final coordinates computed as a weighted average of the top-M matches

## Features

- **Image queries** — geolocate a single photo with optional out-of-range detection via Pair-VPR
- **Video queries** — geolocate a video clip using per-frame voting with geographic clustering and out-of-range detection
- **Adaptive re-ranking** — Pair-VPR only runs when needed, saving compute on confident matches
- **Dashcam/CCTV support** — automatic letterbox detection and configurable HUD crop
- **Area-agnostic** — works for any region with street-level imagery coverage

## Quick Start

### Requirements

- Python 3.10+
- CUDA GPU recommended (CPU works but feature extraction is slow)
- ~5 GB disk for model weights + reference data (varies with area size)

### Setup

```bash
git clone https://github.com/ja1902/VPR.git && cd VPR
./setup.sh
```

The setup script will:
1. Create a virtual environment and install dependencies
2. Download pre-trained weights (~1.8 GB from the official SALAD, DINOv2 and Pair-VPR sources)
3. Ask you for the bounding box of the area you want to cover
4. Build the reference database (harvest panoramas, extract features, build index)

### Query

Activate the virtual environment first:

```bash
source venv/bin/activate
```

**Geolocate a photo:**

```bash
python query.py image photo.jpg
python query.py image photo.jpg --rerank
```

**Geolocate a video:**

```bash
python query.py video clip.mp4
python query.py video clip.mp4 --rerank
python query.py video dashcam.mp4 --rerank --crop 10,25
```

### Options

| Flag | Description |
|------|-------------|
| `--rerank` | Enable adaptive Pair-VPR re-ranking for higher accuracy |
| `--top_k N` | Number of results to return (default: 10) |
| `--fps N` | Video frame sampling rate in Hz (default: 2.0) |
| `--max_frames N` | Maximum frames to extract from video (default: 15) |
| `--crop T,B` | Crop top T% and bottom B% of video frames (for dashcam HUD overlays) |

## Project Structure

```
VPR/
├── setup.sh                  # One-command setup
├── query.py                  # Unified CLI entry point
├── vpr/                      # Core package
│   ├── config.py             # Configuration (loads from .env)
│   ├── model_utils.py        # SALAD model loading
│   ├── pairvpr_utils.py      # Pair-VPR model loading and re-ranking
│   ├── harvest.py            # Download street-level panoramas
│   ├── reproject.py          # Equirectangular → perspective views
│   ├── filter_images.py      # Remove blurry/dark images
│   ├── extract_features.py   # Extract SALAD descriptors
│   ├── build_index.py        # Build FAISS search index
│   ├── query_image.py        # Image query engine
│   └── query_video.py        # Video query engine
├── .env                      # Area coordinates (generated by setup)
├── requirements.txt          # Python dependencies
├── salad/                    # SALAD model (vendored)
├── dinov2/                   # DINOv2 model (vendored)
├── Pair-VPR/                 # Pair-VPR model (vendored)
├── weights/                  # Model weights (downloaded by setup)
└── data/                     # Reference database (built by setup)
    ├── panoramas/            # Raw street-level panoramas
    ├── perspectives/         # Reprojected perspective images
    ├── features/             # SALAD descriptor arrays
    └── index/                # FAISS index + ID map
```

## Configuration

The `.env` file controls the target region:

```env
MIN_LAT=51.435      # South boundary
MAX_LAT=51.475      # North boundary
MIN_LON=-2.615      # West boundary
MAX_LON=-2.545      # East boundary
GRID_STEP_M=20      # Sampling density in meters
```

To cover a different area, edit `.env` and re-run the offline pipeline:

```bash
python -m vpr.harvest
python -m vpr.filter_images
python -m vpr.extract_features
python -m vpr.build_index
```

## Limitations

- Accuracy depends on street-level imagery coverage in the target area
- Out-of-range detection for images requires `--rerank` (Pair-VPR must run); without it, queries outside the reference area will return the nearest in-coverage match
- Video queries work best on continuous footage of one location — clips with scene cuts or unrelated content will produce unreliable results
- Large areas require significant disk space and processing time for the offline pipeline
