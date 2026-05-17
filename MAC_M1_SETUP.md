# Configuration Mac M1 pour PokeScan

## Différences Mac M1 vs Windows

### GPU/Accélération
- **Mac M1**: Utilise Metal Performance Shaders (MPS) via PyTorch
- **Windows**: ROCm pour AMD GPU (RX 6750 XT)
- Le code détecte automatiquement la plateforme et utilise le bon backend

### Virtualenv
Sur Mac M1, créer un seul environnement:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pas besoin des multiples venvs Windows (pokescan-env, pokescan-amdnightly, etc.)

### PyTorch pour Mac M1
```bash
pip install torch torchvision torchaudio
```
PyTorch détecte automatiquement MPS (Metal Performance Shaders).

### EasyOCR et CLIP
Sur Mac M1, pas besoin du sidecar subprocess - EasyOCR et CLIP fonctionnent directement avec MPS.
Le code dans `ocr.py` et `embeddings.py` détecte Windows+ROCm et active le sidecar uniquement là.

## Fichiers à transférer depuis Windows

### 1. Modèles de grading (YOLO classifiers)
Transférer les poids entraînés:
```
runs/classify/high_grade_baseline_3e_amdnightly/weights/best.pt
runs/classify/pair_yolo26n_baseline/weights/best.pt
```

### 2. Datasets (optionnel, si vous voulez réentraîner)
- `dataset_pokemon/` (3.5 GB) - dataset brut scrapé
- `dataset_high_grade/` (1.2 GB) - dataset préparé pour training grades 8/9/10
- `dataset_pairs_high_grade/` (1.2 GB) - dataset front/back pairs

### 3. Index FAISS (optionnel, pour identification visuelle)
Si vous avez construit l'index FAISS:
```
data/faiss_index_*.bin
data/faiss_index_*.json
```
Sinon, reconstruire avec: `python build_card_index.py --languages en fr ja`

### 4. Cache TCGdex (optionnel)
```
data/tcgdex_cache.json
```
Se régénère automatiquement si absent (24h TTL).

## Commandes principales sur Mac M1

### Identification (fonctionne immédiatement)
```bash
source .venv/bin/activate
python api_server.py                    # API server
python identify_card.py --source photo.jpg --lang fr
```

### Grading (nécessite les weights transférés)
```bash
# Entraînement avec MPS
python train_high_grade_classifier.py --epochs 8 --imgsz 384 --batch 16 --device mps

# Inférence
python pokescan_amd.py --source photo.jpg --weights runs/classify/.../weights/best.pt
```

### Build FAISS index
```bash
python build_card_index.py --languages en fr ja
```

## Tests
```bash
pytest                                   # tous les tests
pytest tests/test_identify_card_db.py   # tests identification
```

## Notes importantes

1. **Pas de ROCm sur Mac** - le code détecte automatiquement et utilise MPS
2. **Pas de sidecar subprocess** - EasyOCR/CLIP fonctionnent directement
3. **Performance MPS** - généralement plus rapide que CPU, mais différent de ROCm
4. **Expo app** - mettre à jour `API_URL` dans `pokescan-app/App.tsx` avec l'IP de votre Mac

## Transfert recommandé

**Minimum pour continuer le grading:**
```bash
# Sur Windows, créer une archive
tar -czf pokescan_models.tar.gz \
  runs/classify/high_grade_baseline_3e_amdnightly/weights/best.pt \
  runs/classify/pair_yolo26n_baseline/weights/best.pt

# Transférer vers Mac et extraire
tar -xzf pokescan_models.tar.gz
```

**Complet (avec datasets):**
```bash
# Sur Windows
tar -czf pokescan_full.tar.gz \
  runs/classify/*/weights/best.pt \
  dataset_pokemon/ \
  dataset_high_grade/ \
  dataset_pairs_high_grade/

# ~6-7 GB compressé
```
