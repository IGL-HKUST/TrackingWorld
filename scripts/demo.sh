#!/bin/bash
set -e  # Stop immediately if an error occurs

# ================= Path Configuration =================
# Project root directory
CODE_ROOT="."
# Data working directory (folder containing videos)
WORKDIR="${CODE_ROOT}/demo_data"

# Check if the working directory exists
if [ ! -d "$WORKDIR" ]; then
    echo "Error: WORKDIR does not exist: $WORKDIR"
    exit 1
fi

echo "========================================"
echo "Starting Pipeline"
echo "Code Root: $CODE_ROOT"
echo "Work Dir:  $WORKDIR"
echo "========================================"

# ================= 1. Preprocess: RAM & GPT =================
echo "[Step 1/6] Running RAM & GPT Preprocessing..."
python "${CODE_ROOT}/scripts/ram_gpt.py" \
    --workdir "$WORKDIR" \

# ================= 2. Depth Estimation: UniDepth =================
echo "[Step 2/6] Running UniDepth..."
python "${CODE_ROOT}/scripts/run_unidepth.py" \
    --workdir "$WORKDIR" \
    --v2

# ================= 3. Dense Tracking: EFEP =================
echo "[Step 3/6] Running Dense Tracking (EFEP)..."
python "${CODE_ROOT}/scripts/run_efep.py" \
    --video_path "$WORKDIR" \
    --output_path "$WORKDIR" \
    --use_fp16

# ================= 4. Segmentation: DINO & SAM2 =================
echo "[Step 4/6] Running DINO & SAM2..."
python "${CODE_ROOT}/scripts/run_dino_sam2.py" \
    --workdir "$WORKDIR"

# ================= 5. Video Segmentation: DEVA =================
echo "[Step 5/6] Running DEVA..."
python "${CODE_ROOT}/submodules/Tracking-Anything-with-DEVA/evaluation/eval_with_detections.py" \
    --workdir "$WORKDIR" \
    --model "${CODE_ROOT}/checkpoints/DEVA-propagation.pth"

# ================= 6. Reconstruction: Uni4D =================
echo "[Step 6/6] Running Uni4D Demo..."
# Note: Illegal spaces from the original command have been cleaned
python "${CODE_ROOT}/uni4d/uni4d/run.py" \
    --config "${CODE_ROOT}/uni4d/uni4d/config/config.yaml" \
    --experiment_name base\
    --workdir "$WORKDIR" \
    --cotracker_path densetrack3d_efep \
    --global_downsample_rate 8 \
    --depth_version v2 \
    --num_BA_epochs 1000 \
    --gpu '0' \
    --num_motion_start_epochs 300 \
    --optimize_dyn_upsample \
    --save_upsample

echo "========================================"
echo "Pipeline Finished Successfully!"
echo "========================================"