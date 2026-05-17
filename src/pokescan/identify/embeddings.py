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
import os
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

# Lazy-loaded heavy dependencies.
_model = None
_index = None
_card_ids: list[str] = []
_card_languages: list[str | None] = []

INDEX_FILENAME = "card_embeddings.index"
META_FILENAME = "card_embeddings_meta.json"
EMBEDDING_DIM = 512  # CLIP ViT-B/32


def _is_windows_rocm_runtime() -> bool:
    if sys.platform != "win32":
        return False
    try:
        return "+rocm" in metadata.version("torch").lower()
    except metadata.PackageNotFoundError:
        return False


def _sidecar_python_command() -> list[str]:
    configured = os.environ.get("POKESCAN_CPU_PYTHON")
    if configured:
        return [configured]
    if sys.platform == "win32":
        return ["py", "-3.11"]
    return [sys.executable]


def _visual_search_sidecar(
    image: np.ndarray,
    top_k: int,
) -> list[tuple[str, float, str | None]] | None:
    if os.environ.get("POKESCAN_VISUAL_SIDECAR_ACTIVE") == "1":
        return None

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp, image)
        image_path = tmp.name

    script = r"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("src"))
from pokescan.identify.embeddings import visual_search_detailed

image = np.load(sys.argv[1])
top_k = int(sys.argv[2])
print(json.dumps(visual_search_detailed(image, top_k=top_k), ensure_ascii=False))
"""
    env = os.environ.copy()
    env["POKESCAN_VISUAL_SIDECAR_ACTIVE"] = "1"
    try:
        completed = subprocess.run(
            [*_sidecar_python_command(), "-c", script, image_path, str(top_k)],
            cwd=Path(__file__).resolve().parents[3],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )
        data = json.loads(completed.stdout.strip().splitlines()[-1])
        return [(str(card_id), float(score), lang) for card_id, score, lang in data]
    except Exception:
        return None
    finally:
        try:
            Path(image_path).unlink()
        except OSError:
            pass


def _get_device() -> str:
    """Detect best available device for CLIP inference."""
    forced_device = os.environ.get("POKESCAN_CLIP_DEVICE")
    if forced_device:
        print(f"  [CLIP] Device forcé: {forced_device}")
        return forced_device

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
    global _index, _card_ids, _card_languages

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
        _card_languages = meta.get("card_languages", [None] * len(_card_ids))
        if len(_card_languages) != len(_card_ids):
            _card_languages = [None] * len(_card_ids)
        return _index.ntotal > 0 and len(_card_ids) == _index.ntotal
    except Exception:
        _index = None
        _card_ids = []
        _card_languages = []
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


def encode_images(images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    """Encode BGR card images into normalized CLIP embedding vectors."""
    import cv2
    from PIL import Image as PILImage

    if not images:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    pil_images = [
        PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for image in images
    ]

    model = _get_model()
    embeddings = model.encode(
        pil_images,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings),
        where=norms > 0,
    )

    return embeddings.astype(np.float32)


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
    return [(card_id, score) for card_id, score, _lang in visual_search_detailed(image, top_k)]


def visual_search_detailed(
    image: np.ndarray,
    top_k: int = 10,
) -> list[tuple[str, float, str | None]]:
    """Search the FAISS index and include the indexed card language.

    Older index metadata did not store languages, so the language can be None.
    """
    global _index, _card_ids, _card_languages

    if _is_windows_rocm_runtime():
        sidecar_results = _visual_search_sidecar(image, top_k)
        if sidecar_results is not None:
            return sidecar_results

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
        language = _card_languages[idx] if idx < len(_card_languages) else None
        results.append((_card_ids[idx], score, language))

    return results


def is_index_available() -> bool:
    """Check if a FAISS index exists and can be loaded."""
    global _index
    if _index is not None:
        return _index.ntotal > 0 and len(_card_ids) == _index.ntotal
    return _load_index()


def is_index_file_present() -> bool:
    """Check whether index files exist, without importing FAISS."""
    index_dir = _get_index_dir()
    return (index_dir / INDEX_FILENAME).exists() and (index_dir / META_FILENAME).exists()


def get_index_stats() -> dict[str, Any]:
    """Return stats about the current index."""
    if _index is None:
        if not _load_index():
            stats: dict[str, Any] = {
                "available": False,
                "files_present": is_index_file_present(),
                "total_cards": 0,
            }
            try:
                import faiss  # noqa: F401
            except ImportError as exc:
                stats["reason"] = f"FAISS import failed: {exc}"
            return stats

    return {
        "available": True,
        "total_cards": _index.ntotal if _index else 0,
        "dimension": EMBEDDING_DIM,
        "card_ids_count": len(_card_ids),
        "card_languages_count": len(_card_languages),
    }


# ---------------------------------------------------------------------------
# Index building utilities (used by build_card_index.py)
# ---------------------------------------------------------------------------

def build_index(
    card_images: list[tuple[str, np.ndarray] | tuple[str, str, np.ndarray]],
    output_dir: Path | None = None,
) -> int:
    """Build a FAISS index from a list of (card_id, image) pairs.

    Args:
        card_images: List of (card_id, BGR image) or (card_id, language, BGR image)
            tuples.
        output_dir: Where to save. Defaults to data/embeddings/.

    Returns:
        Number of cards indexed.
    """
    if not card_images:
        return 0

    card_embeddings = []

    for item in card_images:
        if len(item) == 2:
            card_id, image = item
            language = None
        else:
            card_id, language, image = item
        try:
            emb = encode_image(image)
            card_embeddings.append((card_id, language, emb))
        except Exception:
            continue

    return save_index(card_embeddings, output_dir)


def save_index(
    card_embeddings: list[tuple[str, str | None, np.ndarray]],
    output_dir: Path | None = None,
) -> int:
    """Save a FAISS index from precomputed CLIP embeddings.

    Args:
        card_embeddings: List of (card_id, language, normalized vector) tuples.
        output_dir: Where to save. Defaults to data/embeddings/.

    Returns:
        Number of cards indexed.
    """
    import faiss

    if not card_embeddings:
        return 0

    output_dir = output_dir or _get_index_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    card_ids = [card_id for card_id, _language, _emb in card_embeddings]
    card_languages = [language for _card_id, language, _emb in card_embeddings]
    embeddings = [emb for _card_id, _language, emb in card_embeddings]

    matrix = np.stack(embeddings).astype(np.float32)
    if matrix.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected embeddings with dimension {EMBEDDING_DIM}, got {matrix.shape[1]}",
        )

    # Use IndexFlatIP (inner product = cosine similarity on normalized vectors).
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(matrix)

    faiss.write_index(index, str(output_dir / INDEX_FILENAME))

    meta = {
        "card_ids": card_ids,
        "card_languages": card_languages,
        "total": len(card_ids),
        "dimension": EMBEDDING_DIM,
    }
    (output_dir / META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # Reset cached index so next search reloads.
    global _index, _card_ids, _card_languages
    _index = None
    _card_ids = []
    _card_languages = []

    return len(card_ids)


def load_saved_embeddings(
    index_dir: Path | None = None,
) -> list[tuple[str, str | None, np.ndarray]]:
    """Load stored FAISS vectors and metadata for resuming an index build."""
    import faiss

    index_dir = index_dir or _get_index_dir()
    index_path = index_dir / INDEX_FILENAME
    meta_path = index_dir / META_FILENAME

    if not index_path.exists() or not meta_path.exists():
        return []

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    card_ids = meta.get("card_ids", [])
    card_languages = meta.get("card_languages", [None] * len(card_ids))
    if len(card_languages) != len(card_ids):
        card_languages = [None] * len(card_ids)
    if len(card_ids) != index.ntotal:
        raise ValueError(
            f"Index has {index.ntotal} vectors but metadata has {len(card_ids)} cards",
        )

    vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
    index.reconstruct_n(0, index.ntotal, vectors)
    return [
        (card_id, language, vector)
        for card_id, language, vector in zip(card_ids, card_languages, vectors)
    ]
