import json
import sys

import numpy as np

sys.path.insert(0, "src")

from pokescan.identify import embeddings


def test_visual_index_keeps_indexed_language(tmp_path, monkeypatch):
    monkeypatch.setenv("POKESCAN_VISUAL_SIDECAR_ACTIVE", "1")
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    vector = np.zeros(embeddings.EMBEDDING_DIM, dtype=np.float32)
    vector[0] = 1.0

    monkeypatch.setattr(embeddings, "encode_image", lambda _image: vector)
    monkeypatch.setattr(embeddings, "_get_index_dir", lambda: tmp_path)

    count = embeddings.build_index([("swsh7-169", "fr", image)], output_dir=tmp_path)

    assert count == 1
    meta = json.loads((tmp_path / embeddings.META_FILENAME).read_text(encoding="utf-8"))
    assert meta["card_ids"] == ["swsh7-169"]
    assert meta["card_languages"] == ["fr"]
    assert embeddings.visual_search(image) == [("swsh7-169", 1.0)]
    assert embeddings.visual_search_detailed(image) == [("swsh7-169", 1.0, "fr")]


def test_visual_index_supports_legacy_metadata_without_languages(tmp_path, monkeypatch):
    monkeypatch.setenv("POKESCAN_VISUAL_SIDECAR_ACTIVE", "1")
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    vector = np.zeros(embeddings.EMBEDDING_DIM, dtype=np.float32)
    vector[0] = 1.0

    monkeypatch.setattr(embeddings, "encode_image", lambda _image: vector)
    monkeypatch.setattr(embeddings, "_get_index_dir", lambda: tmp_path)

    embeddings.build_index([("swsh7-169", "fr", image)], output_dir=tmp_path)
    meta_path = tmp_path / embeddings.META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["card_languages"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    embeddings._index = None
    embeddings._card_ids = []
    embeddings._card_languages = []

    assert embeddings.visual_search_detailed(image) == [("swsh7-169", 1.0, None)]
