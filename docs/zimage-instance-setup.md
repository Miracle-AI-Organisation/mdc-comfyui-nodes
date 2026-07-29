# Z-Image instance provisioning

Everything a fresh Z-Image (`realistic_z`) ComfyUI box needs. Verified against a
live box: ComfyUI `0.24.0` at `/workspace/ComfyUI`, conda Python 3.11 at
`/opt/conda` (no venv), torch `2.8.0+cu128`, launched with
`main.py --listen 0.0.0.0 --port 8188 --disable-smart-memory`.

## 1. Pin torch before anything else

A custom node's `requirements.txt` can silently downgrade torch and break the
GPU. Snapshot the working versions and constrain every later install to them.

```bash
pip freeze | grep -iE '^(torch|torchvision|torchaudio)==' | tee /workspace/pip-constraints.txt
```

## 2. Install the node packs

The Z-Image workflows need exactly three non-core packs. Everything else they
use is core ComfyUI.

```bash
cd /workspace/ComfyUI/custom_nodes \
 && git clone https://github.com/Miracle-AI-Organisation/mdc-comfyui-nodes.git \
 && git clone https://github.com/tsogzark/ComfyUI-load-image-from-url.git \
 && git clone https://github.com/rgthree/rgthree-comfy.git
```

| Pack | Provides | Used by |
|---|---|---|
| `mdc-comfyui-nodes` | `MdcFaceCrop` | faceswap graph |
| `ComfyUI-load-image-from-url` | `LoadImageFromUrlOrPath` | faceswap graph |
| `rgthree-comfy` | `Power Lora Loader (rgthree)` | both graphs |

**Impact Pack / Impact Subpack are no longer required.** They only existed for
the `UltralyticsDetectorProvider` → `BboxDetectorSEGS` →
`ImpactSEGSOrderedFilter` → `SEGSToImageList` crop chain that `MdcFaceCrop`
replaced. Leaving them installed is harmless; a new box does not need them.

## 3. Install requirements (torch-constrained)

`mdc-comfyui-nodes` has **no** requirements: OpenCV already ships with ComfyUI
and the 227 KB YuNet model lives in the repo. The other packs may have some, so
install whatever is present — always with the constraint file from step 1.

```bash
for d in /workspace/ComfyUI/custom_nodes/*/; do [ -f "$d/requirements.txt" ] && pip install -c /workspace/pip-constraints.txt -r "$d/requirements.txt"; done
```

## 4. Verify the environment before restarting

```bash
python -c "import cv2, torch; print('cv2', cv2.__version__, '| YuNet:', hasattr(cv2, 'FaceDetectorYN')); print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

`YuNet: True` requires OpenCV >= 4.5.4. `cuda True` confirms step 1 protected torch.

## 5. Restart ComfyUI

A new custom node is only picked up by a fresh process.

```bash
pkill -f "main.py --listen"; cd /workspace/ComfyUI && nohup python main.py --listen 0.0.0.0 --port 8188 --disable-smart-memory > /workspace/comfy.log 2>&1 &
```

## 6. Confirm the node registered

```bash
sleep 25; curl -s 127.0.0.1:8188/object_info/MdcFaceCrop | head -c 300; echo; grep -iE 'error|traceback|mdc' /workspace/comfy.log | tail -20
```

An empty `{}` from `/object_info` means the node failed to import — the reason
is in `comfy_log`, most often an OpenCV older than 4.5.4.

## Updating the node later

```bash
git -C /workspace/ComfyUI/custom_nodes/mdc-comfyui-nodes pull && pkill -f "main.py --listen"
```

(then relaunch as in step 5)

## Regenerating this list from a known-good box

If a box drifts, dump its authoritative pack list and compare:

```bash
for d in /workspace/ComfyUI/custom_nodes/*/; do printf '%-45s %s\n' "$(basename "$d")" "$(git -C "$d" remote get-url origin 2>/dev/null)"; done
```
