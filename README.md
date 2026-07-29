# mdc-comfyui-nodes

Custom ComfyUI nodes for the Dream Companion image pipeline.

## `MDC Face Crop (YuNet)`

Detects one face and returns it as a cropped `IMAGE`. Built for the Z-Image
(`realistic_z`) faceswap workflow, where the crop is used as the identity
reference for `ReferenceLatent`.

**Inputs**

| Input | Default | Notes |
|---|---|---|
| `image` | — | Any `IMAGE`. |
| `crop_factor` | `2.0` | Box expansion around the face. `1.0` = the bare face box. |
| `conf_threshold` | `0.6` | YuNet score threshold. |
| `selection` | `confidence` | `confidence` or `largest`. Prefer the default — see below. |
| `fallback` | `full_image` | No face found: pass the full image through, or `error` to fail loudly. |

**Outputs:** `image` (crop), `face_found` (BOOLEAN), `info` (STRING with box size and confidence).

### Why this exists

It replaces this chain:

```
UltralyticsDetectorProvider -> BboxDetectorSEGS -> ImpactSEGSOrderedFilter -> SEGSToImageList
```

Two problems with that chain in production:

1. **The detector silently rots.** `UltralyticsDetectorProvider`'s output is
   cached by ComfyUI's execution cache, so a YOLO model holding CUDA tensors
   stays resident in the process indefinitely. It is not a `ModelPatcher`, so
   ComfyUI's memory manager does not track it; under VRAM pressure its weights
   get freed or overwritten and the node keeps "succeeding" while returning
   boxes over random background. Observed on a live box: a 832x1216 portrait
   whose face is 297x410 produced 8x18 boxes over a blank wall. Restarting
   ComfyUI — or merely switching to a different model file, which forces a
   fresh load — makes it work again for a while.
2. **Ranking detections by area amplifies errors.** `ImpactSEGSOrderedFilter`
   with `target = area(=w*h)` keeps the *biggest* box, so a single oversized
   false positive always outranks the real face.

YuNet runs inside OpenCV's own DNN engine on the CPU: no CUDA allocation, so
there is nothing for ComfyUI's memory manager to reclaim, and a fresh detector
is created per call (`FaceDetectorYN.create` is sub-millisecond). Selection is
by confidence, and a miss degrades to the full frame instead of producing an
empty output that crashes the graph downstream.

### Install

```bash
git clone https://github.com/Miracle-AI-Organisation/mdc-comfyui-nodes.git /workspace/ComfyUI/custom_nodes/mdc-comfyui-nodes
```

Restart ComfyUI. No pip installs and no model downloads: OpenCV is already a
ComfyUI dependency, and the 227 KB YuNet model
(`models/face_detection_yunet_2023mar.onnx`, from the
[OpenCV Zoo](https://github.com/opencv/opencv_zoo)) ships in this repo.

Requires OpenCV >= 4.5.4 for `cv2.FaceDetectorYN`.

### Tests

Pure geometry (no OpenCV or torch needed):

```bash
python3 tests/test_geometry.py
```
