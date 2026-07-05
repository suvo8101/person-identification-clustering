"""
Groups face embeddings into per-identity clusters and assigns each face a
confidence score describing how certain the system is that the face belongs
to the identity it was grouped under.

Algorithm: DBSCAN over the 128-d face_recognition embeddings, using
Euclidean distance (the metric face_recognition's model is trained/tuned
for; ~0.6 is the library's own suggested "same person" threshold, we
default a bit tighter at 0.45 to bias toward precision over recall for a
demo dataset).

Why DBSCAN over k-means:
  - We don't know the number of distinct people ahead of time.
  - It naturally leaves genuinely unique faces as their own single-member
    cluster instead of forcing them into a nearby-but-wrong group.
  - It's robust to clusters of very different sizes (someone with 20
    photos vs. someone with 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.cluster import DBSCAN

from .pipeline import FaceRecord

DEFAULT_EPS = 0.45
DEFAULT_MIN_SAMPLES = 1


@dataclass
class ClusteredFace:
    record: FaceRecord
    cluster_id: int
    confidence: float
    is_singleton: bool


def _confidence_from_distance(distance: float, eps: float) -> float:
    """
    Maps an intra-cluster distance to an intuitive 0-100 confidence score.

    distance == 0        -> 100% (identical embedding / same crop)
    distance == eps       -> ~50% (right at the boundary we clustered on)
    distance >> eps       -> approaches 0%

    We use a smooth exponential decay rather than a hard linear cutoff so
    scores don't cluster unnaturally at the edges.
    """
    score = 100.0 * np.exp(-1.386 * (distance / eps))  # ln(4) ≈ 1.386 -> f(eps)=50
    return float(np.clip(score, 0.0, 100.0))


def cluster_faces(
    face_records: List[FaceRecord],
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> List[ClusteredFace]:
    if not face_records:
        return []

    encodings = np.stack([f.encoding for f in face_records])

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    raw_labels = db.fit_predict(encodings)

    # DBSCAN can mark low-density points as noise (-1). For a person-ID
    # task we still want to show that face -- just as its own singleton
    # identity -- rather than discarding it, so we remap each noise point
    # to a brand new cluster id of its own.
    next_new_id = int(raw_labels.max()) + 1 if raw_labels.size else 0
    labels = raw_labels.copy()
    for i, lbl in enumerate(labels):
        if lbl == -1:
            labels[i] = next_new_id
            next_new_id += 1

    # Compute a centroid per cluster, then score each member by its
    # distance to that centroid.
    centroids: Dict[int, np.ndarray] = {}
    for lbl in set(labels.tolist()):
        members = encodings[labels == lbl]
        centroids[lbl] = members.mean(axis=0)

    cluster_sizes = {lbl: int((labels == lbl).sum()) for lbl in set(labels.tolist())}

    results: List[ClusteredFace] = []
    for record, lbl, encoding in zip(face_records, labels, encodings):
        lbl = int(lbl)
        is_singleton = cluster_sizes[lbl] == 1
        if is_singleton:
            # Nothing to compare against -- it's definitionally "itself".
            confidence = 100.0
        else:
            dist = float(np.linalg.norm(encoding - centroids[lbl]))
            confidence = _confidence_from_distance(dist, eps)

        results.append(
            ClusteredFace(
                record=record,
                cluster_id=lbl,
                confidence=round(confidence, 2),
                is_singleton=is_singleton,
            )
        )

    return results
