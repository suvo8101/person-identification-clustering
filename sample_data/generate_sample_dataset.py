"""
Generates a tiny synthetic image set so you can exercise the full
upload -> detect -> cluster -> display pipeline without needing a real
labeled photo set on hand.

IMPORTANT: this draws simple procedurally-varied cartoon "face" shapes,
not real photographic faces -- `face_recognition`'s detector is trained on
real faces and will likely find few or none of these. Its only purpose is
to smoke-test the upload/API/UI wiring end to end. For the actual
submission, run the pipeline against the real `person_identification`
dataset.

Usage:
    python3 generate_sample_dataset.py
"""

import os
import random

from PIL import Image, ImageDraw

OUT_DIR = os.path.join(os.path.dirname(__file__), "generated")
NUM_IDENTITIES = 4
PHOTOS_PER_IDENTITY = (2, 5)  # inclusive random range


def _draw_face(draw, cx, cy, skin, eye_offset, mouth_curve, size=180):
    r = size // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=skin)
    eye_y = cy - r // 4
    for dx in (-eye_offset, eye_offset):
        draw.ellipse([cx + dx - 10, eye_y - 10, cx + dx + 10, eye_y + 10], fill="white")
        draw.ellipse([cx + dx - 4, eye_y - 4, cx + dx + 4, eye_y + 4], fill="black")
    mouth_y = cy + r // 3
    draw.arc(
        [cx - 40, mouth_y - 20 + mouth_curve, cx + 40, mouth_y + 20 + mouth_curve],
        start=20, end=160, fill="black", width=4,
    )


def make_identity_photo(identity_seed, variation_seed, out_path):
    rng = random.Random(f"{identity_seed}-{variation_seed}")
    base_rng = random.Random(identity_seed)  # stays fixed across this identity's photos

    canvas_size = 320
    img = Image.new("RGB", (canvas_size, canvas_size), color=rng.choice(
        ["#20242c", "#2a2118", "#1b2621", "#241b26"]
    ))
    draw = ImageDraw.Draw(img)

    skin = base_rng.choice(["#e8b98a", "#c68863", "#8d5a3c", "#f2d3b3"])
    eye_offset = 28 + base_rng.randint(-3, 3)

    # "lighting/expression/angle" variation per photo, identity stays fixed
    mouth_curve = rng.randint(-10, 10)
    cx = canvas_size // 2 + rng.randint(-15, 15)
    cy = canvas_size // 2 + rng.randint(-15, 15)
    size = 180 + rng.randint(-20, 20)

    _draw_face(draw, cx, cy, skin, eye_offset, mouth_curve, size=size)
    img.save(out_path, quality=90)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for person_idx in range(NUM_IDENTITIES):
        identity_seed = f"person-{person_idx}"
        n_photos = random.Random(identity_seed).randint(*PHOTOS_PER_IDENTITY)
        for shot in range(n_photos):
            filename = f"{identity_seed}_{shot:02d}.jpg"
            make_identity_photo(identity_seed, shot, os.path.join(OUT_DIR, filename))
            total += 1
    print(f"Wrote {total} synthetic images to {OUT_DIR}")
    print("Note: these are cartoon placeholders for wiring smoke-tests only,")
    print("not real faces -- use a real photo set for the actual demo/submission.")


if __name__ == "__main__":
    main()
