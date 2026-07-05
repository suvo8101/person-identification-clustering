"""
Person Identification & Clustering API
=======================================

Flow:
  1. POST /api/sessions            -> create a session, get session_id
  2. POST /api/sessions/{id}/upload -> upload a batch of raw images
  3. POST /api/sessions/{id}/process -> detect faces, embed, cluster
  4. GET  /api/sessions/{id}/results -> fetch clustering results (cached)
  5. GET  /api/images/{id}/{filename} and /api/crops/{id}/{filename}
                                      -> serve images for the frontend

Run with:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .clustering import DEFAULT_EPS, DEFAULT_MIN_SAMPLES, cluster_faces
from .pipeline import SUPPORTED_EXTENSIONS, extract_faces
from .schemas import (
    BoundingBox,
    ClusterResult,
    FaceResult,
    HealthResponse,
    ProcessResponse,
    UploadResponse,
)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(
    title="Person Identification & Clustering API",
    description="Detects faces across an unorganized photo set and groups them by identity.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo/assessment scope; lock this down for production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _session_dir(session_id: str) -> str:
    return os.path.join(UPLOADS_DIR, session_id)


def _crops_dir(session_id: str) -> str:
    return os.path.join(RESULTS_DIR, session_id, "crops")


def _results_path(session_id: str) -> str:
    return os.path.join(RESULTS_DIR, session_id, "results.json")


def _require_session(session_id: str) -> str:
    path = _session_dir(session_id)
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail=f"Unknown session_id '{session_id}'")
    return path


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=app.version)


@app.post("/api/sessions", response_model=dict)
def create_session() -> dict:
    session_id = uuid.uuid4().hex[:10]
    os.makedirs(_session_dir(session_id), exist_ok=True)
    return {"session_id": session_id}


@app.post("/api/sessions/{session_id}/upload", response_model=UploadResponse)
async def upload_images(session_id: str, files: List[UploadFile] = File(...)) -> UploadResponse:
    session_dir = _require_session(session_id)

    accepted, rejected = [], []
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            rejected.append(f.filename or "unknown")
            continue
        dest = os.path.join(session_dir, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        accepted.append(f.filename)

    total_images = len([f for f in os.listdir(session_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS])

    return UploadResponse(
        session_id=session_id,
        accepted_files=accepted,
        rejected_files=rejected,
        total_images=total_images,
    )


@app.post("/api/sessions/{session_id}/process", response_model=ProcessResponse)
def process_session(
    session_id: str,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> ProcessResponse:
    session_dir = _require_session(session_id)

    image_files = [f for f in os.listdir(session_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
    if not image_files:
        raise HTTPException(status_code=400, detail="No images uploaded for this session yet.")

    crops_dir = _crops_dir(session_id)
    face_records, images_with_no_face = extract_faces(session_dir, crops_dir)

    if not face_records:
        raise HTTPException(status_code=422, detail="No faces were detected in any uploaded image.")

    clustered = cluster_faces(face_records, eps=eps, min_samples=min_samples)

    # Group into clusters, ordered by size (largest identity groups first).
    by_cluster: dict[int, list] = {}
    for cf in clustered:
        by_cluster.setdefault(cf.cluster_id, []).append(cf)

    def to_face_result(cf) -> FaceResult:
        r = cf.record
        top, right, bottom, left = r.bbox
        return FaceResult(
            face_id=r.face_id,
            image_filename=r.image_filename,
            image_url=f"/api/images/{session_id}/{r.image_filename}",
            crop_url=f"/api/crops/{session_id}/{r.face_id}.jpg",
            bbox=BoundingBox(top=top, right=right, bottom=bottom, left=left),
            cluster_id=cf.cluster_id,
            confidence=cf.confidence,
            is_singleton=cf.is_singleton,
        )

    clusters: List[ClusterResult] = []
    ordered_ids = sorted(by_cluster, key=lambda cid: -len(by_cluster[cid]))
    for display_index, cid in enumerate(ordered_ids, start=1):
        members = [to_face_result(cf) for cf in sorted(by_cluster[cid], key=lambda c: -c.confidence)]
        avg_conf = round(sum(m.confidence for m in members) / len(members), 2)
        clusters.append(
            ClusterResult(
                cluster_id=cid,
                label=f"Person {display_index}",
                size=len(members),
                avg_confidence=avg_conf,
                representative_face=members[0],
                members=members,
            )
        )

    response = ProcessResponse(
        session_id=session_id,
        total_images=len(image_files),
        total_faces_detected=len(face_records),
        images_with_no_face=images_with_no_face,
        num_clusters=len(clusters),
        eps=eps,
        min_samples=min_samples,
        clusters=clusters,
    )

    os.makedirs(os.path.dirname(_results_path(session_id)), exist_ok=True)
    with open(_results_path(session_id), "w") as f:
        f.write(response.model_dump_json(indent=2))

    return response


@app.get("/api/sessions/{session_id}/results", response_model=ProcessResponse)
def get_results(session_id: str) -> ProcessResponse:
    path = _results_path(session_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No results yet -- call /process first.")
    with open(path) as f:
        return ProcessResponse(**json.load(f))


@app.get("/api/images/{session_id}/{filename}")
def get_image(session_id: str, filename: str) -> FileResponse:
    path = os.path.join(_session_dir(session_id), filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.get("/api/crops/{session_id}/{filename}")
def get_crop(session_id: str, filename: str) -> FileResponse:
    path = os.path.join(_crops_dir(session_id), filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Crop not found")
    return FileResponse(path)
