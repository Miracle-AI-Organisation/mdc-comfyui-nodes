"""YuNet face detection + crop geometry.

Deliberately free of torch and of ComfyUI imports: the detector runs on the CPU
inside OpenCV (YuNet is an ONNX model executed by cv2's own DNN backend), so its
weights live in host RAM and are invisible to ComfyUI's CUDA memory manager.
That is the whole point of this module — a long-lived detector held in the
ComfyUI process must not be something the VRAM manager can free underneath us.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
# YuNet is trained at 320x320; cv2 rescales internally via setInputSize.
_DEFAULT_INPUT_SIZE = (320, 320)


def bundled_model_path() -> str:
    """Absolute path of the YuNet model shipped alongside this file."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", MODEL_FILENAME)


def expand_and_clamp(
    box: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    crop_factor: float,
) -> Tuple[int, int, int, int]:
    """Expands an (x, y, w, h) box around its centre and clamps it to the image.

    Returns `(x1, y1, x2, y2)`. The result is always inside the image and always
    at least one pixel wide/high, so callers can slice with it unconditionally.
    """
    x, y, w, h = box
    centre_x = x + w / 2.0
    centre_y = y + h / 2.0
    half_w = max(w, 1) * crop_factor / 2.0
    half_h = max(h, 1) * crop_factor / 2.0

    x1 = int(round(centre_x - half_w))
    y1 = int(round(centre_y - half_h))
    x2 = int(round(centre_x + half_w))
    y2 = int(round(centre_y + half_h))

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))
    return x1, y1, x2, y2


class Face:
    """One detection: pixel box, confidence and the 5 YuNet landmarks."""

    __slots__ = ("box", "confidence", "landmarks")

    def __init__(
        self,
        box: Tuple[int, int, int, int],
        confidence: float,
        landmarks: List[Tuple[int, int]],
    ) -> None:
        self.box = box
        self.confidence = confidence
        self.landmarks = landmarks

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Face(box={self.box}, confidence={self.confidence:.3f})"


def detect_faces(
    bgr_image,
    model_path: Optional[str] = None,
    conf_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    top_k: int = 50,
) -> List[Face]:
    """Runs YuNet on a BGR uint8 numpy image, best detection first.

    A fresh detector is created per call on purpose: `FaceDetectorYN.create` is
    sub-millisecond, and holding no state means there is nothing to corrupt
    between generations.
    """
    import cv2  # imported lazily so the geometry helpers stay dependency-free

    path = model_path or bundled_model_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YuNet model not found at {path}")

    height, width = bgr_image.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        model=path,
        config="",
        input_size=_DEFAULT_INPUT_SIZE,
        score_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
    )
    detector.setInputSize((width, height))

    _, raw = detector.detect(bgr_image)
    if raw is None:
        return []

    faces: List[Face] = []
    for row in raw:
        x, y, w, h = (int(round(v)) for v in row[0:4])
        landmarks = [
            (int(round(row[4 + i * 2])), int(round(row[5 + i * 2]))) for i in range(5)
        ]
        faces.append(Face((x, y, w, h), float(row[14]), landmarks))

    faces.sort(key=lambda f: f.confidence, reverse=True)
    return faces


def pick_face(faces: List[Face], strategy: str = "confidence") -> Optional[Face]:
    """Chooses one detection.

    `confidence` (default) is deliberately NOT `largest`: ranking by area lets a
    single oversized false positive outrank the real face, which is exactly how
    the previous YOLO-based chain produced background crops.
    """
    if not faces:
        return None
    if strategy == "largest":
        return max(faces, key=lambda f: f.area)
    return max(faces, key=lambda f: f.confidence)


def crop_face(
    bgr_image,
    crop_factor: float = 1.6,
    conf_threshold: float = 0.6,
    strategy: str = "confidence",
    model_path: Optional[str] = None,
):
    """Detects and crops the chosen face.

    Returns `(cropped_bgr, face_or_None)`. When no face is found the **input
    image is returned unchanged** — fail-open, so a detector miss degrades the
    identity reference instead of failing the whole generation.
    """
    height, width = bgr_image.shape[:2]
    faces = detect_faces(
        bgr_image, model_path=model_path, conf_threshold=conf_threshold
    )
    face = pick_face(faces, strategy)
    if face is None:
        return bgr_image, None

    x1, y1, x2, y2 = expand_and_clamp(face.box, width, height, crop_factor)
    return bgr_image[y1:y2, x1:x2], face
