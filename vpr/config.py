from pathlib import Path
import math
import os

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PANORAMA_DIR = DATA_DIR / "panoramas"
PERSPECTIVE_DIR = DATA_DIR / "perspectives"
FEATURES_DIR = DATA_DIR / "features"
INDEX_DIR = DATA_DIR / "index"

# Load region coordinates from .env file
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

_missing = [k for k in ("MIN_LAT", "MAX_LAT", "MIN_LON", "MAX_LON") if not os.environ.get(k)]
if _missing:
    raise RuntimeError(
        f"Region not configured ({', '.join(_missing)} unset). "
        "Run ./setup.sh or copy .env.example to .env and fill in the bounding box."
    )

MIN_LAT = float(os.environ["MIN_LAT"])
MAX_LAT = float(os.environ["MAX_LAT"])
MIN_LON = float(os.environ["MIN_LON"])
MAX_LON = float(os.environ["MAX_LON"])

METERS_PER_DEGREE_LAT = 111_320
METERS_PER_DEGREE_LON = METERS_PER_DEGREE_LAT * math.cos(math.radians((MIN_LAT + MAX_LAT) / 2))
GRID_STEP_M = int(os.environ.get("GRID_STEP_M", "20"))
LAT_STEP = GRID_STEP_M / METERS_PER_DEGREE_LAT
LON_STEP = GRID_STEP_M / METERS_PER_DEGREE_LON

# Harvesting
PANORAMA_ZOOM = 4
MAX_CONCURRENT_REQUESTS = 5
FIND_RADIUS_M = 25

# Reprojection
PERSPECTIVE_FOV = 90
PERSPECTIVE_SIZE = 322

# Filtering
BLUR_THRESHOLD = 100.0
DARKNESS_THRESHOLD = 30

# SALAD feature extraction
SALAD_MODEL = "serizba/salad"
SALAD_VARIANT = "dinov2_salad"
SALAD_INPUT_SIZE = 322
SALAD_DIM = 8448
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))

# Run model weights/activations in float16 on CUDA (halves VRAM, ~2x faster on T4-class GPUs)
USE_FP16 = os.environ.get("USE_FP16", "1") == "1"

# Pair-VPR re-ranking
PAIRVPR_CANDIDATES = 100
PAIRVPR_BATCH_SIZE = int(os.environ.get("PAIRVPR_BATCH_SIZE", "16"))

# Video query
VIDEO_SAMPLE_FPS = 2.0
VIDEO_MAX_FRAMES = 15

# FAISS query
TOP_K = 10

# FAISS index. The index holds compressed vectors (PCA + 8-bit scalar quantization)
# and only produces a shortlist; the shortlist is re-scored exactly against the full
# descriptors, which stay on disk (memory-mapped). Set INDEX_FACTORY to override the
# automatic choice, e.g. "Flat", "SQ8", "PCA1024,SQ8", "PCA2048,IVF2048,SQ8".
INDEX_FACTORY = os.environ.get("INDEX_FACTORY", "")
INDEX_PCA_DIM = int(os.environ.get("INDEX_PCA_DIM", "2048"))
INDEX_IVF_MIN_VECTORS = 2_000_000  # exhaustive scan is ~2 GB at 1M vectors; IVF trades recall for speed
INDEX_NPROBE = int(os.environ.get("INDEX_NPROBE", "64"))
INDEX_TRAIN_SAMPLES = 50_000
INDEX_ADD_CHUNK = 20_000
SHORTLIST_K = int(os.environ.get("SHORTLIST_K", "1024"))

METADATA_FILE = DATA_DIR / "metadata.json"
DESCRIPTORS_FILE = FEATURES_DIR / "descriptors.npy"
INDEX_FILE = INDEX_DIR / "region.index"
REFERENCE_DB_FILE = INDEX_DIR / "references.sqlite"
