#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  Visual Place Recognition — Setup"
echo "========================================"
echo

# ── 1. Check Python ──────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ is required but not found."
    exit 1
fi
echo "[1/6] Python: $($PYTHON --version)"

# Check CUDA
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "       CUDA: available"
else
    echo "       CUDA: not detected (CPU mode — feature extraction will be slow)"
fi

# ── 2. Virtual environment ───────────────────────────────────────
if [ ! -d "venv" ]; then
    echo
    echo "[2/6] Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo
    echo "[2/6] Virtual environment already exists."
fi

source venv/bin/activate
pip install --upgrade pip -q

echo "       Installing dependencies..."
pip install -r requirements.txt -q
echo "       Done."

# ── 3. Download model weights ────────────────────────────────────
# Model repos (salad, dinov2, Pair-VPR) are vendored in the repository.
# Weights are downloaded from each model's official source.
SALAD_URL="https://github.com/serizba/salad/releases/download/v1.0.0"
DINOV2_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14"
PAIRVPR_URL="https://huggingface.co/CSIRORobotics/Pair-VPR/resolve/main"

echo
echo "[3/6] Downloading model weights (~1.8 GB total)..."
mkdir -p weights

download_weight() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "       $(basename "$dest") already exists — skipping."
    else
        echo "       Downloading $(basename "$dest")..."
        curl -L -o "$dest" "$url"
    fi
}

download_weight "$SALAD_URL/dino_salad.ckpt"                     "weights/dino_salad.ckpt"
download_weight "$DINOV2_URL/dinov2_vitb14_pretrain.pth"         "weights/dinov2_vitb14_pretrain.pth"
download_weight "$DINOV2_URL/dinov2_vitb14_reg4_pretrain.pth"    "weights/dinov2_vitb14_reg4_pretrain.pth"
download_weight "$PAIRVPR_URL/pairvpr-vitB.pth"                  "weights/pairvpr-vitB.pth"

# ── 4. Configure target region ───────────────────────────────────
echo
echo "[4/6] Configure target region"

if [ -f ".env" ]; then
    echo "       .env already exists:"
    cat .env
    echo
    read -p "       Reconfigure? (y/N): " reconfig
    if [ "$reconfig" != "y" ] && [ "$reconfig" != "Y" ]; then
        echo "       Keeping existing configuration."
        SKIP_CONFIG=1
    fi
fi

if [ -z "$SKIP_CONFIG" ]; then
    echo
    echo "  Enter the bounding box for the area you want to cover."
    echo "  Use decimal degrees (WGS84). Example for a city centre:"
    echo "    SW corner: 51.435, -2.615"
    echo "    NE corner: 51.475, -2.545"
    echo

    read -p "  Min latitude  (south boundary): " MIN_LAT
    read -p "  Max latitude  (north boundary): " MAX_LAT
    read -p "  Min longitude (west boundary):  " MIN_LON
    read -p "  Max longitude (east boundary):  " MAX_LON
    read -p "  Grid step in meters [20]:        " GRID_STEP_M
    GRID_STEP_M=${GRID_STEP_M:-20}

    cat > .env <<EOF
MIN_LAT=$MIN_LAT
MAX_LAT=$MAX_LAT
MIN_LON=$MIN_LON
MAX_LON=$MAX_LON
GRID_STEP_M=$GRID_STEP_M
EOF

    echo
    echo "  Saved to .env"
fi

# ── 5. Build reference database ──────────────────────────────────
echo
echo "[5/6] Building reference database (this may take hours for large areas)..."
echo

echo "  Step 1/4: Harvesting panoramas and reprojecting to perspectives..."
python -m vpr.harvest

echo
echo "  Step 2/4: Filtering blurry and dark images..."
python -m vpr.filter_images

echo
echo "  Step 3/4: Extracting SALAD features (GPU recommended)..."
python -m vpr.extract_features

echo
echo "  Step 4/4: Building FAISS search index..."
python -m vpr.build_index

# ── 6. Done ──────────────────────────────────────────────────────
echo
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo
echo "  Query with an image:"
echo "    python query.py image <photo.jpg> --rerank"
echo
echo "  Query with a video:"
echo "    python query.py video <clip.mp4> --rerank"
echo
