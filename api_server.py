#!/usr/bin/env python3
"""PokeScan API Server — FastAPI backend for card identification.

Usage:
    uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
    python api_server.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="PokeScan API",
    description="API d'identification de cartes Pokémon — OCR + TCGdex",
    version="1.0.0",
)

# Allow all origins for development (Expo dev client needs this).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _warmup_models() -> None:
    """Warm OCR and visual models after startup so first mobile request is faster."""
    try:
        import numpy as np

        from pokescan.identify.ocr import extract_card_text

        dummy = np.zeros((896, 640, 3), dtype=np.uint8)
        extract_card_text(dummy)
        print("warmup ocr=ok", flush=True)

        if os.environ.get("POKESCAN_WARMUP_VISUAL", "1") != "0":
            from pokescan.identify.embeddings import is_index_available, visual_search_detailed

            if is_index_available():
                visual_search_detailed(dummy, top_k=1)
                print("warmup visual=ok", flush=True)
    except Exception as exc:
        print(f"warmup error={type(exc).__name__}: {exc}", flush=True)


@app.on_event("startup")
async def startup_warmup() -> None:
    threading.Thread(target=_warmup_models, daemon=True).start()


def _candidate_to_dict(c) -> dict:
    return {
        "card_id": c.card_id,
        "name": c.name,
        "set_name": c.set_name,
        "set_id": c.set_id,
        "number": c.number,
        "number_total": c.number_total,
        "rarity": c.rarity,
        "language": c.language,
        "hp": c.hp,
        "image_url": c.image_url,
        "pricing": c.pricing,
        "score": round(c.score, 4),
        "score_detail": {k: round(v, 4) for k, v in c.score_detail.items()},
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "pokescan-api"}


@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
    language: str = Form("fr"),
    top_k: int = Form(5),
    skip_crop: bool = Form(False),
    visual_mode: str = Form("auto"),
):
    """Identify a Pokémon card from an uploaded image.

    Returns OCR results, best match, and top-K candidates with pricing.
    """
    from pokescan.identify.pipeline import identify_card_from_bytes

    started = time.perf_counter()
    contents = await file.read()
    if not contents:
        return JSONResponse(
            status_code=400,
            content={"error": "Image vide"},
        )

    debug_dir = ROOT / "data" / "debug_requests"
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "latest_upload.jpg").write_bytes(contents)

    result = identify_card_from_bytes(
        contents,
        language=language,
        top_k=top_k,
        skip_crop=skip_crop,
        visual_mode=visual_mode,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    used_visual = any("visual" in c.score_detail for c in result.candidates)

    response = {
        "ocr": {
            "name": result.ocr_result.name,
            "collector_number": result.ocr_result.collector_number,
            "hp": result.ocr_result.hp,
            "language": result.ocr_result.language,
            "year": result.ocr_result.year,
            "raw_texts": result.ocr_result.raw_texts,
        },
        "confidence": round(result.confidence, 4),
        "best_match": _candidate_to_dict(result.best_match) if result.best_match else None,
        "candidates": [_candidate_to_dict(c) for c in result.candidates],
        "debug": {
            "elapsed_ms": elapsed_ms,
            "used_visual": used_visual,
            "visual_mode": visual_mode,
        },
    }
    (debug_dir / "latest_response.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    best = result.best_match.card_id if result.best_match else None
    print(
        "identify",
        f"elapsed_ms={elapsed_ms}",
        f"visual={used_visual}",
        f"ocr_name={result.ocr_result.name!r}",
        f"ocr_num={result.ocr_result.collector_number!r}",
        f"best={best!r}",
        flush=True,
    )

    return JSONResponse(content=response)


@app.get("/search")
async def search_card(name: str, language: str = "fr"):
    """Search TCGdex by name (no image required)."""
    from pokescan.identify.card_db import search_by_name

    results = search_by_name(name, language)
    return [
        {
            "card_id": c.card_id,
            "name": c.name,
            "number": c.number,
            "image_url": c.image_url,
        }
        for c in results[:20]
    ]


if __name__ == "__main__":
    import uvicorn

    print("Démarrage du serveur PokeScan API...")
    print("Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
