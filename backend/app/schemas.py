"""Pydantic models describing the shape of API responses."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    session_id: str
    accepted_files: List[str]
    rejected_files: List[str]
    total_images: int


class BoundingBox(BaseModel):
    top: int
    right: int
    bottom: int
    left: int


class FaceResult(BaseModel):
    face_id: str
    image_filename: str
    image_url: str
    crop_url: str
    bbox: BoundingBox
    cluster_id: int
    confidence: float  # 0-100, certainty this face belongs to its assigned cluster
    is_singleton: bool  # True if this is the only face found for this identity


class ClusterResult(BaseModel):
    cluster_id: int
    label: str  # human-friendly label, e.g. "Person 1"
    size: int
    avg_confidence: float
    representative_face: FaceResult
    members: List[FaceResult]


class ProcessResponse(BaseModel):
    session_id: str
    total_images: int
    total_faces_detected: int
    images_with_no_face: List[str]
    num_clusters: int
    eps: float
    min_samples: int
    clusters: List[ClusterResult]


class HealthResponse(BaseModel):
    status: str
    version: str
