
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

YUNET_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = ROOT / "models" / "face_recognition_sface_2021dec.onnx"


def load_models():
    """Load the pretrained YuNet and SFace models."""

    if not YUNET_PATH.exists():
        raise FileNotFoundError(f"YuNet model missing: {YUNET_PATH}")

    if not SFACE_PATH.exists():
        raise FileNotFoundError(f"SFace model missing: {SFACE_PATH}")

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH),
        "",
        (320, 320),
        0.8,
        0.3,
        5000
    )

    recognizer = cv2.FaceRecognizerSF.create(
        str(SFACE_PATH),
        ""
    )

    return detector, recognizer


def extract_embeddings(image, detector, recognizer):
    """Detect faces and extract one normalized embedding per face."""

    if image is None or image.size == 0:
        return [], []

    height, width = image.shape[:2]
    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None:
        return [], []

    embeddings = []
    valid_faces = []

    for face in faces:

        aligned = recognizer.alignCrop(image, face)
        feature = recognizer.feature(aligned).flatten()

        norm = np.linalg.norm(feature)

        if norm == 0:
            continue

        feature = feature / norm

        embeddings.append(feature.astype(np.float32))
        valid_faces.append(face)

    return embeddings, valid_faces


def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between normalized embeddings."""

    a = np.asarray(embedding1, dtype=np.float32)
    b = np.asarray(embedding2, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return -1.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def recognize_image(image, database, detector, recognizer,
                    threshold=0.363):
    """Recognize all detected faces in an image."""

    output = image.copy()
    embeddings, faces = extract_embeddings(
        image, detector, recognizer
    )

    results = []

    for embedding, face in zip(embeddings, faces):

        best_name = "Unknown"
        best_score = -1.0

        for name, stored_embedding in database.items():

            score = cosine_similarity(
                embedding,
                stored_embedding
            )

            if score > best_score:
                best_score = score
                best_name = name

        if best_score < threshold:
            best_name = "Unknown"

        x, y, w, h = face[:4].astype(int)

        color = (
            (0, 180, 0)
            if best_name != "Unknown"
            else (0, 0, 255)
        )

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        label = f"{best_name} | {best_score:.2f}"

        cv2.putText(
            output,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        results.append({
            "Identity": best_name,
            "Similarity": round(best_score, 4)
        })

    return output, results
