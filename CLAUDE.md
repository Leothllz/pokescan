# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project scope

PokeScan is a local prototype with two complementary pillars. Don't mix them up — they share `src/pokescan/` but solve different problems:

1. **Card grading** (original focus): scrape graded card photos → build datasets of front/back pairs → train a YOLO/Ultralytics classifier to estimate grades 8/9/10. Lives in `scrapper.py`, `prepare_*_dataset.py`, `train_high_grade_classifier.py`, `pokescan_amd.py`, `check_dataset.py`.
2. **Card identification** (newer, drives the mobile app): given a phone photo, identify the card via OCR + TCGdex lookup + optional CLIP visual reranking. Lives in `src/pokescan/identify/`, `api_server.py`, `identify_card.py`, `build_card_index.py`, and the Expo client in `pokescan-app/`.

The README is in French and most code comments / log messages are French. Match that style when editing — don't translate existing strings unless asked.

## Common commands

All Python commands assume Windows PowerShell with `pokescan-env\Scripts\Activate.ps1` already activated unless noted. Forward-slash paths still work under Git Bash.

```powershell
# Tests
pytest                                                # full suite
pytest tests/test_identify_card_db.py                 # one file
pytest tests/test_identify_card_db.py::test_search_tcgdex_does_not_fallback_to_english_for_french

# Identification CLI
python identify_card.py --source photo.jpg --lang fr --verbose
python identify_card.py --source folder/ --output results.json

# API server (consumed by the Expo app)
python api_server.py                                  # http://localhost:8000, /docs available
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

# Build the FAISS visual index (one-time, ~30 min for full catalog)
python build_card_index.py --languages en ja fr
python build_card_index.py --languages en --max-per-set 50 --resume   # quick test / resume

# Dataset pipeline (grading side)
python scrapper.py --sources all --max 300
python check_dataset.py
python prepare_card_crops.py --previews --overwrite
python prepare_training_dataset.py --output dataset_high_grade --grades 8,9,10 --tighten --tight-inset 0.01 --overwrite
python prepare_paired_dataset.py --output dataset_pairs_high_grade --grades 8,9,10 --tighten --tight-inset 0.01 --overwrite
python prepare_pair_composite_dataset.py --input dataset_pairs_high_grade --output dataset_pair_composite_high_grade --overwrite

# Training (use the ROCm-capable venv for GPU)
.\pokescan-amdnightly\Scripts\python.exe train_high_grade_classifier.py --epochs 8 --imgsz 384 --batch 16

# Expo mobile client
cd pokescan-app && npm install && npm run start         # then 'a' for android, 'i' for iOS, 'w' for web
```

## Architecture

### Identification pipeline (`src/pokescan/identify/`)

The pipeline in `pipeline.py:identify_card` runs in this order:

1. **Crop**: if the input doesn't already look like a card (aspect ratio 0.62–0.80), `card_crop.find_card_crop` tries saturation + edge heuristics, then falls back to a phone-photo bbox.
2. **Resize**: downscale to `POKESCAN_IDENTIFY_MAX_DIM` (default 1400) longest side.
3. **OCR** (`ocr.py`): EasyOCR reads three zones — top (name), top-right (HP), bottom (collector number + year). Detected language overrides the requested `language` param.
4. **DB search** (`card_db.py`): TCGdex REST API, file-cached in `data/tcgdex_cache.json` (24h TTL). Language is honored strictly — `search_tcgdex` does NOT fall back to English when the OCR language is French (that's a regression guard with a test).
5. **Score fusion** (`matcher.py`): weighted score combining name/number/HP/year/set_total similarity. With visual reranking enabled the weights shift toward CLIP (see `WEIGHTS_OCR_ONLY` vs `WEIGHTS_WITH_VISUAL`).
6. **Visual reranking** (`embeddings.py`, opt-in): `visual_mode="auto"` runs CLIP only when OCR already produced candidates; `"always"` always runs; `"off"` (default) skips entirely. Visual is intentionally not a fallback for failed OCR because it's too noisy on photographed cards.

A strong OCR match (`local_id` + score ≥ 0.78, or HP + score ≥ 0.55) short-circuits visual reranking regardless of `visual_mode` other than `"always"`.

### Windows ROCm sidecar pattern

`ocr.py` and `embeddings.py` both detect a Windows + torch-with-`+rocm` runtime and, when found, transparently delegate EasyOCR / CLIP work to a subprocess running a CPU Python (`POKESCAN_CPU_PYTHON`, default `py -3.11`). The sidecar guards itself against recursion via `POKESCAN_*_SIDECAR_ACTIVE=1`. If you add another heavy torch-based step, follow the same pattern — Windows ROCm torch currently can't run EasyOCR/CLIP locally due to driver limits, and the rest of the codebase relies on this being invisible.

### API → app boundary

`api_server.py` is FastAPI with permissive CORS for the Expo dev client. The startup hook warms up OCR (and optionally CLIP via `POKESCAN_WARMUP_VISUAL`) on a daemon thread so the first mobile request isn't 20 s slow. `/identify` writes the latest upload + response to `data/debug_requests/` for inspection. `pokescan-app/App.tsx` has a hard-coded `API_URL` near the top (currently `http://192.168.1.150:8000`) — update it for your network when testing the mobile client.

### Source layout

`src/pokescan/` is imported via `sys.path.insert(0, "src")` at the top of every CLI/server entrypoint — there is no `pip install -e .`. `pokescan.paths` is the single source of truth for filesystem locations (`DATA_DIR`, `WEIGHTS_DIR`, `SCRAPED_DATASET_DIR`, `YOLO_*`); reuse those constants instead of hard-coding paths.

### Multiple virtualenvs by design

`pokescan-env/` (vanilla Windows CPU), `pokescan-amdnightly/` (Windows ROCm nightly for `gfx103X-dgpu`), `pokescan-rocm312/`, `pokescan-rocmtest/`, and `.venv-wsl/` (WSL2 ROCm) all coexist intentionally — different hardware paths need different torch builds. Don't suggest consolidating them. The grading-training commands explicitly use `.\pokescan-amdnightly\Scripts\python.exe`; the identification stack runs on `pokescan-env`.

`HIP_VISIBLE_DEVICES` must be **unset** for the RX 6750 XT under Windows ROCm — setting it masks the GPU. See `setup_amd.md` for the WSL2 spoofing path (`HSA_OVERRIDE_GFX_VERSION=10.3.0` for the gfx1031 → gfx1030 trick).

## Grading dataset analysis

`dataset_pokemon/` contains **6,839 images** (front/back pairs) across grades 1-10:

```
Grade 10:  1,457 images (21%)  - Pristine cards
Grade 9.5: 2,163 images (32%)  - Near-mint+
Grade 9:   2,306 images (34%)  - Near-mint
Grade 8:     643 images (9%)   - Excellent
Grade 7:     137 images (2%)   - Near-mint-
Grade 6-1:   103 images (1.5%) - Lower grades (very sparse)
```

**Current approach**: Training only on grades 8/9/10 (high-grade classifier) ignores 87% of available grade diversity.

**Recommendations for improvement**:

1. **Multi-class classifier**: Train on all grades with class weights to handle imbalance. The dataset has enough 7-10 samples for meaningful learning.

2. **Grouped classifier**: Simplify to 3 classes to balance dataset:
   - Low (1-6): 103 images
   - Mid (7-8): 780 images  
   - High (8.5-10): 5,939 images

3. **Defect detection approach**: Instead of global grade classification, detect specific defects:
   - Corner wear (YOLO object detection)
   - Centering measurement (border ratio analysis)
   - Surface scratches (texture analysis)
   - Combine defects → grade estimate with explainability

4. **Data augmentation**: Target grades 1-7 with rotation, flip, brightness variations to balance the dataset.

**Fundamental limitation**: Models trained on photos of graded cards learn photo quality more than card quality. For production use, would need controlled photo setup and defect-based features rather than end-to-end classification.

## What's gitignored and why it matters

`data/` (FAISS index + TCGdex cache), `dataset*/`, `weights/`, `runs/`, all venvs, and `*.ipynb` are ignored. If a function references something in those paths, it's expected to be regenerated by the relevant command — don't commit them and don't expect them to exist on a fresh checkout. `build_card_index.py --resume` and the TCGdex cache regenerate themselves; the grading datasets are rebuilt from `dataset_pokemon/` via the `prepare_*` scripts.
