"""
Face extraction pipeline.

Responsible for turning a folder of raw, unorganized images into a flat list
of "face records" -- one entry per detected face, each carrying:
  - which image it came from
  - where in the image it was found (bounding box)
  - a 128-d embedding describing the face, used later for clustering

We use `face_recognition` (a thin wrapper around dlib's ResNet face
recognition model). It is CPU-friendly and gives solid accuracy for a
take-home-sized dataset without needing a GPU.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import List

import face_recognition
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# "hog" is fast and CPU-only (good default for an assessment/demo).
# Switch to "cnn" if a GPU is available and higher recall is needed on
# hard poses/angles.
DETECTION_MODEL = os.environ.get("FACE_DETECTION_MODEL", "hog")
UPSAMPLE_TIMES = int(os.environ.get("FACE_UPSAMPLE", "1"))


@dataclass
class FaceRecord:
    face_id: str
    image_filename: str
    image_path: str
    bbox: tuple  # (top, right, bottom, left) - face_recognition convention
    encoding: np.ndarray
    crop_path: str = ""


def _is_supported(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS


def _save_crop(image: Image.Image, bbox: tuple, out_path: str, pad_ratio: float = 0.35) -> None:
    """Crop the face out of the source image with a small margin so the
    thumbnail shown in the UI includes some hair/context, not just a
    landmark-tight box."""
    top, right, bottom, left = bbox
    h = bottom - top
    w = right - left
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)

    img_w, img_h = image.size
    top = max(0, top - pad_h)
    left = max(0, left - pad_w)
    bottom = min(img_h, bottom + pad_h)
    right = min(img_w, right + pad_w)

    image.crop((left, top, right, bottom)).save(out_path, quality=90)


def extract_faces(session_dir: str, crops_dir: str) -> tuple[List[FaceRecord], List[str]]:
    """
    Walk every image in `session_dir`, detect all faces, and compute an
    embedding for each. Face crops are written to `crops_dir` for the
    frontend to display as thumbnails.

    Returns (face_records, images_with_no_face).
    """
    os.makedirs(crops_dir, exist_ok=True)

    face_records: List[FaceRecord] = []
    images_with_no_face: List[str] = []

    filenames = sorted(f for f in os.listdir(session_dir) if _is_supported(f))

    for filename in filenames:
        image_path = os.path.join(session_dir, filename)
        try:
            image = face_recognition.load_image_file(image_path)
        except Exception:
            images_with_no_face.append(filename)
            continue

        locations = face_recognition.face_locations(
            image, number_of_times_to_upsample=UPSAMPLE_TIMES, model=DETECTION_MODEL
        )
        if not locations:
            images_with_no_face.append(filename)
            continue

        encodings = face_recognition.face_encodings(image, known_face_locations=locations)

        pil_image = Image.fromarray(image)
        for bbox, encoding in zip(locations, encodings):
            face_id = uuid.uuid4().hex[:12]
            crop_path = os.path.join(crops_dir, f"{face_id}.jpg")
            _save_crop(pil_image, bbox, crop_path)

            face_records.append(
                FaceRecord(
                    face_id=face_id,
                    image_filename=filename,
                    image_path=image_path,
                    bbox=bbox,
                    encoding=encoding,
                    crop_path=crop_path,
                )
            )

    return face_records, images_with_no_face
