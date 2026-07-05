"""
Unit tests for the clustering/confidence logic in isolation from the face
detection model (so they run fast, deterministically, and without needing
dlib/face_recognition installed).
"""

import numpy as np
import pytest

from app.clustering import cluster_faces
from app.pipeline import FaceRecord


def _make_record(face_id, filename, encoding):
    return FaceRecord(
        face_id=face_id,
        image_filename=filename,
        image_path=f"/fake/{filename}",
        bbox=(0, 100, 100, 0),
        encoding=encoding,
    )


@pytest.fixture
def synthetic_records():
    """
    Two well-separated 'people' (A: 4 photos, B: 3 photos) plus one
    person who only appears once (C), mimicking real face_recognition
    embedding statistics (128-d, unit-ish scale, small intra-person
    variance from lighting/angle/expression).
    """
    rng = np.random.default_rng(42)
    identity_a = rng.normal(0, 1, 128)
    identity_a /= np.linalg.norm(identity_a)
    identity_b = rng.normal(0, 1, 128)
    identity_b /= np.linalg.norm(identity_b)
    identity_c = rng.normal(0, 1, 128)
    identity_c /= np.linalg.norm(identity_c)

    records = []
    for i in range(4):
        e = identity_a + rng.normal(0, 0.015, 128)
        records.append(_make_record(f"a{i}", f"imgA{i}.jpg", e))
    for i in range(3):
        e = identity_b + rng.normal(0, 0.015, 128)
        records.append(_make_record(f"b{i}", f"imgB{i}.jpg", e))
    records.append(_make_record("c0", "imgC0.jpg", identity_c))
    return records


def test_same_person_photos_land_in_one_cluster(synthetic_records):
    clustered = cluster_faces(synthetic_records, eps=0.45, min_samples=1)

    by_filename_prefix = {}
    for cf in clustered:
        prefix = cf.record.image_filename[3]  # 'A', 'B', or 'C'
        by_filename_prefix.setdefault(prefix, set()).add(cf.cluster_id)

    assert len(by_filename_prefix["A"]) == 1, "All of A's photos should share one cluster id"
    assert len(by_filename_prefix["B"]) == 1, "All of B's photos should share one cluster id"
    assert by_filename_prefix["A"] != by_filename_prefix["B"], "A and B must not be merged"


def test_unique_person_becomes_a_confident_singleton(synthetic_records):
    clustered = cluster_faces(synthetic_records, eps=0.45, min_samples=1)
    c_face = next(cf for cf in clustered if cf.record.image_filename == "imgC0.jpg")

    assert c_face.is_singleton is True
    assert c_face.confidence == 100.0


def test_confidence_scores_are_bounded_and_reasonable(synthetic_records):
    clustered = cluster_faces(synthetic_records, eps=0.45, min_samples=1)
    for cf in clustered:
        assert 0.0 <= cf.confidence <= 100.0
    # Tightly clustered synthetic photos should score as high-confidence matches.
    non_singleton_scores = [cf.confidence for cf in clustered if not cf.is_singleton]
    assert min(non_singleton_scores) > 50.0


def test_empty_input_returns_empty_list():
    assert cluster_faces([], eps=0.45, min_samples=1) == []
