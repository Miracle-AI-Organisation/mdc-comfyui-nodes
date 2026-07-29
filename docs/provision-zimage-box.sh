# ============ 0. config ============
export COMFY=/workspace/ComfyUI
export CIVITAI_TOKEN="<YOUR_CIVITAI_TOKEN>"

# ============ 1. ComfyUI (skip clone if present) + venv that INHERITS system torch ============
[ -d "$COMFY/.git" ] || git clone https://github.com/comfyanonymous/ComfyUI "$COMFY"
# rm -rf /workspace/venv
python3 -m venv /workspace/venv --system-site-packages      # inherits existing torch
V=/workspace/venv/bin
$V/python -m pip install --upgrade pip

# ============ 2. verify inherited torch is 2.8.0+cu128 (NO download if good) ============
$V/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
# Expect: torch 2.8.0+cu128 cuda 12.8 avail True  -> reused, skip installing torch.
# ONLY if wrong version / avail False:
#   $V/pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# ============ 3. ComfyUI core requirements ============
$V/pip install -r "$COMFY/requirements.txt"

# ============ 4. custom node packs (all 6) ============
cd "$COMFY/custom_nodes"
[ -d mdc-comfyui-nodes ]              || git clone https://github.com/Miracle-AI-Organisation/mdc-comfyui-nodes
[ -d ComfyUI-Impact-Pack ]            || git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack
[ -d ComfyUI-Impact-Subpack ]         || git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack
[ -d ComfyUI-SeedVR2_VideoUpscaler ]  || git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
[ -d rgthree-comfy ]                  || git clone https://github.com/rgthree/rgthree-comfy
[ -d ComfyUI-load-image-from-url ]    || git clone https://github.com/tsogzark/ComfyUI-load-image-from-url
for r in mdc-comfyui-nodes ComfyUI-Impact-Pack ComfyUI-Impact-Subpack ComfyUI-SeedVR2_VideoUpscaler rgthree-comfy ComfyUI-load-image-from-url; do
  [ -f "$r/requirements.txt" ] && $V/pip install -r "$r/requirements.txt"
done
# mdc-comfyui-nodes has no requirements.txt on purpose: OpenCV already ships with
# ComfyUI and the 227 KB YuNet model is committed in the repo.

# ============ 5. opencv numpy-ABI pin ============
$V/pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
$V/pip install "opencv-python-headless>=4.10,<4.14"
# MdcFaceCrop needs cv2.FaceDetectorYN (OpenCV >= 4.5.4) — check it now, not at generation time:
$V/python -c "import cv2; print('cv2', cv2.__version__, 'YuNet', hasattr(cv2, 'FaceDetectorYN'))"

# ============ 5b. safety: confirm node installs didn't drift torch off cu128 ============
$V/python -c "import torch; print('torch now', torch.__version__, torch.version.cuda)"
# If NOT 2.8.0+cu128:
#   $V/pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# ============ 6. models ============
mkdir -p "$COMFY/models/diffusion_models" "$COMFY/models/text_encoders" "$COMFY/models/vae" "$COMFY/models/ultralytics/bbox"

# text encoders (HF, public)
wget -O "$COMFY/models/text_encoders/qwen_3_8b_fp8mixed.safetensors" "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"
wget -O "$COMFY/models/text_encoders/qwen_3_4b.safetensors"          "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"

# VAEs (HF, public)
wget -O "$COMFY/models/vae/flux2-vae.safetensors" "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"
wget -O "$COMFY/models/vae/ae.safetensors"        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"

# face detectors (HF, public) — NO LONGER USED by the Z-Image workflows (MdcFaceCrop
# replaced the Ultralytics chain). Keep only for other workflows that need Impact detectors.
wget -O "$COMFY/models/ultralytics/bbox/face_yolov8m.pt" "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt"
wget -O "$COMFY/models/ultralytics/bbox/face_yolov9c.pt" "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov9c.pt"

# diffusion checkpoints (civitai — renamed to generic names the workflow expects)
curl -sSL -o "$COMFY/models/diffusion_models/edit-checkpoint.safetensors" "https://civitai.com/api/download/models/2740209?fileId=2626634&token=$CIVITAI_TOKEN"
wget -O "$COMFY/models/diffusion_models/zit-checkpoint.safetensors" "https://civitai.red/api/download/models/3025713?token=$CIVITAI_TOKEN"

# ============ 7. launch (single line) ============
cd "$COMFY" && setsid $V/python main.py --listen 0.0.0.0 --port 8188 --disable-smart-memory > /workspace/comfy_start.log 2>&1 < /dev/null &
