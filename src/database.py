
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "dataset" / "face_database.json"


def load_database():
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not DATABASE_PATH.exists():
        return {}

    with open(DATABASE_PATH, "r") as file:
        data = json.load(file)

    return {
        name: np.asarray(vector, dtype=np.float32)
        for name, vector in data.items()
    }


def save_database(database):
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    serializable_data = {
        name: np.asarray(vector).tolist()
        for name, vector in database.items()
    }

    with open(DATABASE_PATH, "w") as file:
        json.dump(serializable_data, file, indent=4)


def register_embedding(name, embeddings, database):
    """Register one identity using multiple reference images."""

    if not embeddings:
        raise ValueError("No valid face embeddings were provided.")

    mean_embedding = np.mean(
        np.asarray(embeddings),
        axis=0
    )

    norm = np.linalg.norm(mean_embedding)

    if norm == 0:
        raise ValueError("Invalid average embedding.")

    database[name] = mean_embedding / norm

    save_database(database)

    return database


def delete_person(name, database):
    if name in database:
        del database[name]
        save_database(database)

    return database
