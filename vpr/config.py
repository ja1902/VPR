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

MIN_LAT = float(os.environ.get("MIN_LAT", "51.435"))
MAX_LAT = float(os.environ.get("MAX_LAT", "51.475"))
MIN_LON = float(os.environ.get("MIN_LON", "-2.615"))
MAX_LON = float(os.environ.get("MAX_LON", "-2.545"))

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
BATCH_SIZE = 32

# Video query
VIDEO_SAMPLE_FPS = 2.0
VIDEO_MAX_FRAMES = 15

# FAISS query
TOP_K = 10

METADATA_FILE = DATA_DIR / "metadata.json"
