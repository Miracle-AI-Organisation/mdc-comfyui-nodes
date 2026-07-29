#!/usr/bin/env bash
# Provision a Z-Image (realistic_z) ComfyUI box.
#
# Idempotent: re-running skips existing clones and already-downloaded models.
# Verified shape of a live box: ComfyUI 0.24.0 at /workspace/ComfyUI, venv at
# /workspace/venv inheriting system torch 2.8.0+cu128, launched with
# --listen 0.0.0.0 --port 8188 --disable-smart-memory.

set -uo pipefail   # NOT -e: several steps are probes whose failure we handle inline

# ============ 0. config ============
export COMFY="${COMFY:-/workspace/ComfyUI}"
V="${V:-/workspace/venv/bin}"
PINS="${PINS:-/workspace/pip-constraints.txt}"
# Pass the token in the environment rather than editing this file, so it never
# lands in git or in the shell history of the box:
#   CIVITAI_TOKEN=xxxx bash provision-zimage-box.sh
: "${CIVITAI_TOKEN:?export CIVITAI_TOKEN=<your civitai token> before running}"

# Verified download: skips if already present, fails loudly on error pages
# (civitai/HF return a few KB of HTML/JSON when a token is wrong — that would
# otherwise be saved as a .safetensors and only explode at model load time).
fetch() {  # fetch <url> <dest> [min_bytes]
  local url="$1" dest="$2" min="${3:-1000000}" sz
  if [ -f "$dest" ] && [ "$(stat -c%s "$dest")" -ge "$min" ]; then
    echo "skip  $(basename "$dest") (already present)"; return 0
  fi
  echo "get   $(basename "$dest")"
  curl -fL --retry 3 --retry-delay 5 --progress-bar -o "$dest" "$url" || {
    echo "FAILED $url"; return 1; }
  sz=$(stat -c%s "$dest")
  if [ "$sz" -lt "$min" ]; then
    echo "TOO SMALL ($sz bytes) — looks like an error page, not a model:"; head -c 200 "$dest"; echo
    rm -f "$dest"; return 1
  fi
  echo "ok    $(basename "$dest") $sz bytes"
}

# ============ 1. ComfyUI + venv that INHERITS system torch ============
[ -d "$COMFY/.git" ] || git clone https://github.com/comfyanonymous/ComfyUI "$COMFY"
# rm -rf /workspace/venv
[ -x "$V/python" ] || python3 -m venv /workspace/venv --system-site-packages
$V/python -m pip install --upgrade pip

# ============ 2. verify inherited torch is 2.8.0+cu128 (NO download if good) ============
$V/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
# Expect: torch 2.8.0+cu128 cuda 12.8 avail True  -> reused, skip installing torch.
# ONLY if wrong version / avail False:
#   $V/pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# ============ 2b. pin torch so nothing below can move it ============
# Impact-Subpack pulls ultralytics, SeedVR2 pulls torch-adjacent deps, and even
# ComfyUI's own requirements.txt can bump torch. Constrain every install below
# to the versions we just verified: a conflicting pack now fails loudly instead
# of silently breaking CUDA.
$V/pip freeze | grep -iE '^(torch|torchvision|torchaudio)==' | tee "$PINS"

# ============ 3. ComfyUI core requirements ============
$V/pip install -c "$PINS" -r "$COMFY/requirements.txt"

# ============ 4. custom node packs (6) ============
cd "$COMFY/custom_nodes"
[ -d mdc-comfyui-nodes ]              || git clone https://github.com/Miracle-AI-Organisation/mdc-comfyui-nodes
[ -d rgthree-comfy ]                  || git clone https://github.com/rgthree/rgthree-comfy
[ -d ComfyUI-load-image-from-url ]    || git clone https://github.com/tsogzark/ComfyUI-load-image-from-url
[ -d ComfyUI-SeedVR2_VideoUpscaler ]  || git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
# Impact Pack/Subpack: no longer used by the Z-Image workflows (MdcFaceCrop
# replaced the UltralyticsDetectorProvider -> BboxDetectorSEGS -> SEGS chain).
# Keep only if this box also serves workflows that use Impact nodes.
[ -d ComfyUI-Impact-Pack ]            || git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack
[ -d ComfyUI-Impact-Subpack ]         || git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack

# `if` rather than `[ -f ] && ...`: the && form returns non-zero for packs
# without a requirements.txt (ours, rgthree, load-image-from-url), which aborts
# the whole script under `set -e`.
for r in mdc-comfyui-nodes rgthree-comfy ComfyUI-load-image-from-url ComfyUI-SeedVR2_VideoUpscaler ComfyUI-Impact-Pack ComfyUI-Impact-Subpack; do
  if [ -f "$r/requirements.txt" ]; then
    echo "== requirements: $r"
    $V/pip install -c "$PINS" -r "$r/requirements.txt"
  fi
done

# ============ 5. opencv numpy-ABI pin (this is what MdcFaceCrop needs) ============
$V/pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
$V/pip install "opencv-python-headless>=4.10,<4.14"
# YuNet lives in cv2 itself (OpenCV >= 4.5.4) — mdc-comfyui-nodes needs no pip
# deps and no model download: the 227 KB face_detection_yunet_2023mar.onnx is
# committed in the repo.
$V/python -c "import cv2; assert hasattr(cv2, 'FaceDetectorYN'), f'OpenCV {cv2.__version__} predates FaceDetectorYN (need >= 4.5.4)'; print('cv2', cv2.__version__, '| YuNet OK')"

# ============ 5b. safety: confirm node installs didn't drift torch off cu128 ============
$V/python -c "import torch; print('torch now', torch.__version__, torch.version.cuda, 'avail', torch.cuda.is_available())"
# If NOT 2.8.0+cu128:
#   $V/pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# ============ 6. models ============
mkdir -p "$COMFY/models/diffusion_models" "$COMFY/models/text_encoders" "$COMFY/models/vae"

# text encoders (HF, public)
fetch "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" "$COMFY/models/text_encoders/qwen_3_8b_fp8mixed.safetensors"
fetch "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"            "$COMFY/models/text_encoders/qwen_3_4b.safetensors"

# VAEs (HF, public)
fetch "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" "$COMFY/models/vae/flux2-vae.safetensors"
fetch "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"    "$COMFY/models/vae/ae.safetensors"

# diffusion checkpoints (civitai — renamed to the generic names the workflow expects)
# NOTE: absolute destinations on purpose. The old script's `wget -O
# zit-checkpoint.safetensors` ran while cwd was $COMFY/custom_nodes, so the main
# Z-Image UNET landed there and UNETLoader could never find it.
fetch "https://civitai.com/api/download/models/2740209?fileId=2626634&token=$CIVITAI_TOKEN" "$COMFY/models/diffusion_models/edit-checkpoint.safetensors"
fetch "https://civitai.red/api/download/models/3025713?token=$CIVITAI_TOKEN"                "$COMFY/models/diffusion_models/zit-checkpoint.safetensors"

# --- OPTIONAL: YOLO face detectors -----------------------------------------
# Only needed by workflows that still use Impact's UltralyticsDetectorProvider.
# The Z-Image faceswap graph no longer does; skip these (~104 MB) on a
# Z-Image-only box.
# mkdir -p "$COMFY/models/ultralytics/bbox"
# fetch "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" "$COMFY/models/ultralytics/bbox/face_yolov8m.pt"
# fetch "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov9c.pt" "$COMFY/models/ultralytics/bbox/face_yolov9c.pt"

# ============ 7. launch ============
pkill -f "main.py --listen" 2>/dev/null; sleep 2
cd "$COMFY" && setsid $V/python main.py --listen 0.0.0.0 --port 8188 --disable-smart-memory > /workspace/comfy_start.log 2>&1 < /dev/null &

# ============ 8. verify the box came up with MdcFaceCrop registered ============
for i in $(seq 1 40); do curl -sf -m 3 127.0.0.1:8188/system_stats >/dev/null && break; sleep 3; done
echo "== MdcFaceCrop in /object_info:"
curl -s 127.0.0.1:8188/object_info/MdcFaceCrop | head -c 300; echo
# `{}` means the node failed to import — the reason is in the log:
grep -iE 'error|traceback|import fail|mdc' /workspace/comfy_start.log | tail -20
echo "== node packs loaded:"; grep -A20 'Import times for custom nodes' /workspace/comfy_start.log | tail -20
