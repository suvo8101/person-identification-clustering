# Contact Sheet — Person Identification & Clustering

Groups unorganized photos of multiple people into per-identity clusters,
handling varied lighting, angles, and expressions, and gives each photo a
confidence score for its identity match.

```
person-clustering/
├── backend/            FastAPI service: face detection, embeddings, clustering
│   ├── app/
│   │   ├── main.py          API routes
│   │   ├── pipeline.py      face detection + embedding extraction
│   │   ├── clustering.py    DBSCAN grouping + confidence scoring
│   │   └── schemas.py       response models
│   ├── tests/test_clustering.py
│   └── requirements.txt
├── frontend/            Static UI ("contact sheet" view of results)
│   ├── index.html / style.css / app.js
├── sample_data/          Script to generate a synthetic test set
└── data/                 Runtime storage (uploads + results), gitignored
```

## How it works

1. **Detect** — every uploaded image is scanned for faces with `face_recognition`
   (a dlib ResNet model). Each face gets a bounding box and a 128-dimension
   embedding vector that describes its features independent of lighting,
   pose, or expression.
2. **Cluster** — all embeddings across the whole batch are grouped with
   **DBSCAN** (Euclidean distance). DBSCAN was chosen over k-means because:
   - the number of distinct people in the batch is unknown up front,
   - it doesn't force a photo into the wrong group just because it's the
     "closest available" bucket — a face with no real match becomes its
     own singleton identity instead,
   - group sizes can be wildly uneven (one person with 20 photos, another
     with 1), which DBSCAN handles natively.
3. **Score confidence** — for every face, we measure its distance to its
   cluster's centroid and convert that into a 0–100% confidence score using
   smooth exponential decay (rather than a hard cutoff), so scores near the
   clustering threshold don't feel arbitrary. A face that is the only member
   of its cluster (a person who appears once) is reported as `100%` / "unique"
   since there's nothing to compare it against — the UI marks these
   distinctly rather than implying a high-confidence match.
4. **Serve** — the FastAPI backend exposes clusters + cropped face thumbnails
   + confidence scores as JSON; the frontend renders them as a "contact
   sheet": one filmstrip row per identity, each frame stamped with its match
   confidence.

## Quick start

### Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

> `dlib` (a `face-recognition` dependency) compiles from source on some
> platforms and needs `cmake` + a C++ toolchain installed first
> (`brew install cmake` on macOS, `apt install cmake build-essential` on
> Ubuntu). If you'd rather avoid the native build, `pip install dlib-binary`
> before installing the rest of `requirements.txt`.

The API is now live at `http://localhost:8000` (interactive docs at
`http://localhost:8000/docs`).

### Frontend

No build step — it's a static page that calls the API directly.

```bash
cd frontend
python3 -m http.server 5500
```

Open `http://localhost:5500`. If your backend isn't on
`http://localhost:8000`, set `window.API_BASE` at the top of `app.js` (or
in a `<script>` tag before `app.js` loads) to point at it.

### Try it with a synthetic dataset

If you don't have a labeled photo set handy, generate one of simple
placeholder "identity" images to exercise the full upload → cluster →
display flow end to end:

```bash
cd sample_data
python3 generate_sample_dataset.py
```

This writes a small folder of images to `sample_data/generated/` that you
can drag into the UI. (It's a synthetic sanity-check, not a substitute for
testing against the real `person_identification` dataset — swap in real
photos for the actual submission demo.)

### Run tests

```bash
cd backend
pip install pytest
pytest tests/ -v
```

The clustering tests run against synthetic embeddings so they're fast and
don't require `dlib`/`face_recognition` to exercise the grouping and
confidence-scoring logic in isolation.

## API reference

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/sessions` | Start a new session, returns `session_id` |
| `POST` | `/api/sessions/{id}/upload` | Multipart-upload a batch of images |
| `POST` | `/api/sessions/{id}/process?eps=0.45&min_samples=1` | Run detection + clustering |
| `GET` | `/api/sessions/{id}/results` | Re-fetch the last computed results |
| `GET` | `/api/images/{id}/{filename}` | Serve an original uploaded image |
| `GET` | `/api/crops/{id}/{filename}` | Serve a cropped face thumbnail |

`eps` is the DBSCAN distance threshold — lower = stricter matching (more,
smaller clusters), higher = looser matching (fewer, larger clusters). The
frontend exposes this as the "match strictness" slider so you can tune
precision vs. recall live against your dataset without redeploying.

### Example response (`POST /process`)

```json
{
  "session_id": "a1b2c3d4e5",
  "total_images": 12,
  "total_faces_detected": 14,
  "images_with_no_face": ["blurry_07.jpg"],
  "num_clusters": 4,
  "eps": 0.45,
  "clusters": [
    {
      "cluster_id": 0,
      "label": "Person 1",
      "size": 5,
      "avg_confidence": 87.3,
      "members": [
        {
          "face_id": "f3a9c1d2b8e0",
          "image_filename": "IMG_0021.jpg",
          "confidence": 91.4,
          "is_singleton": false,
          "bbox": { "top": 40, "right": 220, "bottom": 180, "left": 80 }
        }
      ]
    }
  ]
}
```

## Design notes / trade-offs

- **Detection model**: defaults to `hog` (CPU-only, fast) via
  `FACE_DETECTION_MODEL` env var; switch to `cnn` for better recall on
  extreme angles/occlusion if a GPU is available.
- **Multiple faces per photo**: handled — each detected face is clustered
  independently, so a group photo can contribute to several different
  identity clusters at once.
- **Threshold tuning**: `eps=0.45` is a reasonable general default for the
  `face_recognition` embedding space (its own docs suggest ~0.6 as a loose
  "same person" cutoff); it's intentionally exposed as a live parameter
  since the right value depends on how visually similar your subjects are
  and how much you want to prioritize precision vs. recall.
- **Confidence ≠ probability**: this is a distance-derived certainty score,
  not a calibrated statistical probability. It's meant to help a human
  reviewer quickly spot borderline matches worth double-checking, not as a
  standalone ground truth.
- **Privacy**: uploaded images and derived face crops are stored under
  `data/` per session and are never sent anywhere outside this service.
  There's no auth layer since this is an assessment/demo scope — add one
  before deploying anywhere multi-tenant.

## Possible extensions

- Swap in an ArcFace/InsightFace embedding model for higher accuracy on
  harder poses.
- Persist sessions in a real datastore instead of the filesystem.
- Let a reviewer manually merge/split clusters and feed corrections back
  into threshold tuning.
- Batch/async processing with a job queue for large datasets.

