import numpy as np
import pytest

KEYWORDS = ["kuendigung", "miete", "urheber"]


def fake_embedder(texts: list[str], task_type: str) -> np.ndarray:
    """Bag-of-keywords vectors, so similarity is predictable without Vertex AI."""
    rows = [[text.lower().count(word) + 0.01 for word in KEYWORDS] for text in texts]
    matrix = np.asarray(rows, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


@pytest.fixture
def embedder():
    return fake_embedder
