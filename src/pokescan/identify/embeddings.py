"""Visual embedding search using CLIP + FAISS.

Provides visual reranking for card identification when OCR is insufficient.
The index is built once from official card images (EN + JP + FR) and reused.

Usage:
    # Build the index (one-time, ~30min for full catalog)
    python build_card_index.py --languages en ja fr --max-per-set 500

    # Use in pipeline (automatic when index exists)
    from pokescan.identify.embeddings import visual_search
    results = visual_search(card_image, top_k=10)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Lazy-loaded heavy dependencies.
_model = None
_index = None
_card_ids: list[str] = []

INDEX_FILENAME = "card_embeddings.index"
META_FILENAME = "card_embeddings_meta.json"
EMBEDDING_DIM = 512  # CLIP ViT-B/32


def _get_device() -> str:
    """Detect best available device for CLIP inference."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  [CLIP] GPU détecté: {name}")
            return "cuda"
    except Exception:
        pass
    print("  [CLIP] Mode CPU")
    return "cpu"


def _get_model():
    """Lazy-load the CLIP model on the best available device."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        device = _get_device()
        _model = SentenceTransformer("clip-ViT-B-32", device=device)
    return _model


def _get_index_dir() -> Path:
    from pokescan.paths import DATA_DIR
    return DATA_DIR / "embeddings"


def _load_index() -> bool:
    """Load the FAISS index and metadata from disk."""
    global _index, _card_ids

    index_dir = _get_index_dir()
    index_path = index_dir / INDEX_FILENAME
    meta_path = index_dir / META_FILENAME

    if not index_path.exists() or not meta_path.exists():
        return False

    try:
        import faiss
        _index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _card_ids = meta.get("card_ids", [])
        return _index.ntotal > 0 and len(_card_ids) == _index.ntotal
    except Exception:
        _index = None
        _card_ids = []
        return False


def encode_image(image: np.ndarray) -> np.ndarray:
    """Encode a BGR card image into a CLIP embedding vector.

    Args:
        image: BGR numpy array (OpenCV format).

    Returns:
        Normalized float32 vector of shape (512,).
    """
    import cv2
    from PIL import Image as PILImage

    # Convert BGR → RGB → PIL.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(rgb)

    model = _get_model()
    embedding = model.encode(pil_image, convert_to_numpy=True)

    # Normalize for cosine similarity via inner product.
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.astype(np.float32)


def visual_search(
    image: np.ndarray,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Search the FAISS index for visually similar cards.

    Args:
        image: BGR card image (already cropped).
        top_k: Number of results.

    Returns:
        List of (card_id, similarity_score) tuples, sorted by descending score.
        Returns empty list if index is not available.
    """
    global _index, _card_ids

    # Load index on first call.
    if _index is None:
        if not _load_index():
            return []

    query = encode_image(image).reshape(1, -1)

    import faiss
    distances, indices = _index.search(query, min(top_k, _index.ntotal))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_card_ids):
            continue
        # Inner product on normalized vectors = cosine similarity in [0, 1].
        score = float(max(0.0, min(1.0, dist)))
        results.append((_card_ids[idx], score))

    return results


def is_index_available() -> bool:
    """Check if a FAISS index exists on disk."""
    index_dir = _get_index_dir()
    return (index_dir / INDEX_FILENAME).exists() and (index_dir / META_FILENAME).exists()


def get_index_stats() -> dict[str, Any]:
    """Return stats about the current index."""
    if _index is None:
        if not _load_index():
            return {"available": False, "total_cards": 0}

    return {
        "available": True,
        "total_cards": _index.ntotal if _index else 0,
        "dimension": EMBEDDING_DIM,
        "card_ids_count": len(_card_ids),
    }


# ---------------------------------------------------------------------------
# Index building utilities (used by build_card_index.py)
# ---------------------------------------------------------------------------

def build_index(
    card_images: list[tuple[str, np.ndarray]],
    output_dir: Path | None = None,
) -> int:
    """Build a FAISS index from a list of (card_id, image) pairs.

    Args:
        card_images: List of (card_id, BGR image) tuples.
        output_dir: Where to save. Defaults to data/embeddings/.

    Returns:
        Number of cards indexed.
    """
    import faiss

    if not card_images:
        return 0

    output_dir = output_dir or _get_index_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _get_model()
    card_ids = []
    embeddings = []

    for card_id, image in card_images:
        try:
            emb = encode_image(image)
            embeddings.append(emb)
            card_ids.append(card_id)
        except Exception:
            continue

    if not embeddings:
        return 0

    matrix = np.stack(embeddings).astype(np.float32)

    # Use IndexFlatIP (inner product = cosine similarity on normalized vectors).
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(matrix)

    faiss.write_index(index, str(output_dir / INDEX_FILENAME))

    meta = {
        "card_ids": card_ids,
        "total": len(card_ids),
        "dimension": EMBEDDING_DIM,
    }
    (output_dir / META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # Reset cached index so next search reloads.
    global _index, _card_ids
    _index = None
    _card_ids = []

    return len(card_ids)
