# PokeScan

PokeScan est un prototype local pour collecter des images de cartes Pokemon gradees, verifier le dataset, puis entrainer/evaluer un modele YOLO pour l'analyse de cartes.

Le chantier actuel stabilise le socle: commandes reproductibles, dependances declarees, logs lisibles, validations de chemins, et structure minimale.

## Installation

```powershell
python -m venv pokescan-env
.\pokescan-env\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
```

Pour une RX 6750 XT avec ROCm, utilisez WSL2 et suivez aussi `setup_amd.md` / `setup_wsl2_rocm.sh`. La roue PyTorch ROCm se pose separement selon la version ROCm disponible.

## Environnement ROCm Windows

Un environnement Windows ROCm nightly AMD `gfx103X-dgpu` peut etre active avec:

```powershell
.\pokescan-amdnightly\Scripts\Activate.ps1
.\scripts\setup_amdnightly.ps1
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Points importants:

- `HIP_VISIBLE_DEVICES` doit etre absent, sinon la RX 6750 XT peut etre masquee.
- Les includes MSVC/UCRT sont necessaires pour que MIOpen compile certains kernels, dont BatchNorm.

## Structure

```text
src/pokescan/        Code reutilisable
tests/              Tests legers
scripts/            Scripts ponctuels
dataset_pokemon/    Dataset scrape local, exclu du versioning
dataset/            Dataset YOLO local, exclu du versioning
weights/            Modeles locaux, exclus du versioning
runs/               Sorties Ultralytics, exclues du versioning
```

## Scraping

```powershell
python scrapper.py --help
python scrapper.py --sources gcc --max 100
python scrapper.py --sources all --max 300
```

Le format conserve pour le dataset scrape est:

```text
dataset_pokemon/<grade>/front/*.jpg
dataset_pokemon/<grade>/back/*.jpg
```

## Statistiques dataset

```powershell
python check_dataset.py
```

La commande affiche les comptes par note avec les colonnes `front`, `back`, `other` et `total`.

## Preparation des crops

Pour retirer les labels PSA/PCA et garder uniquement la zone carte avant entrainement:

```powershell
python prepare_card_crops.py --previews --overwrite
```

Les images preparees sont ecrites dans `dataset_crops/`, avec `manifest.csv` et des controles visuels dans `dataset_crops/_previews/`.

Pour verifier une image precise:

```powershell
python prepare_card_crops.py --pattern *img_1778518429_6.jpg --previews --overwrite
```

Pour faire une deuxieme passe plus serree et reduire les bords de plastique:

```powershell
python prepare_card_crops.py --previews --tighten --tight-inset 0.01 --overwrite
```

## Dataset classification normalise

Pour creer un dataset pret pour une baseline de classification:

```powershell
python prepare_training_dataset.py --tighten --tight-inset 0.01 --overwrite
```

La sortie suit le format:

```text
dataset_normalized/
  train/<grade>/*.jpg
  val/<grade>/*.jpg
  test/<grade>/*.jpg
  manifest.csv
```

Pour filtrer les notes utiles au premier modele:

```powershell
python prepare_training_dataset.py --output dataset_high_grade --grades 8,9,10 --tighten --tight-inset 0.01 --overwrite
```

Les images listees dans `dataset_exclude.txt` sont ignorees automatiquement.

## Dataset front/back paire

Pour eviter de donner une mauvaise etiquette a une face seule, le dataset paire garde le front et le back ensemble avec la note finale de la carte:

```powershell
python prepare_paired_dataset.py --output dataset_pairs_high_grade --grades 8,9,10 --tighten --tight-inset 0.01 --overwrite
```

La sortie suit le format:

```text
dataset_pairs_high_grade/
  train/<grade>/<card_id>_front.jpg
  train/<grade>/<card_id>_back.jpg
  val/<grade>/<card_id>_front.jpg
  val/<grade>/<card_id>_back.jpg
  test/<grade>/<card_id>_front.jpg
  test/<grade>/<card_id>_back.jpg
  manifest.csv
```

Le pairing est fait par nom source stable: index voisin front/back et timestamp proche, pas par position dans une liste. Si une image est dans `dataset_exclude.txt`, toute la paire est ignoree.

Pour entrainer YOLO en voyant les deux faces comme un seul exemple:

```powershell
python prepare_pair_composite_dataset.py --input dataset_pairs_high_grade --output dataset_pair_composite_high_grade --overwrite
```

## Controle qualite dataset

Pour generer une planche de controle rapide:

```powershell
python make_dataset_contact_sheet.py --dataset dataset_high_grade --split train --grade 10 --samples 80 --cols 8
```

Si une image est mauvaise, ajoutez son chemin relatif depuis `dataset_pokemon/` dans `dataset_exclude.txt`, puis regenerez `dataset_high_grade`.

## Entrainement, evaluation, prediction

Baseline classification 8/9/10:

```powershell
.\pokescan-amdnightly\Scripts\python.exe train_high_grade_classifier.py --epochs 8 --imgsz 384 --batch 16
```

Baseline YOLO26 sur composites front/back, avec augmentations prudentes pour le grading:

```powershell
.\pokescan-amdnightly\Scripts\python.exe train_high_grade_classifier.py --data dataset_pair_composite_high_grade --model yolo26n-cls.pt --epochs 8 --imgsz 512 --batch 12 --workers 0 --name pair_yolo26n_baseline
```

## Note sur le notebook Kaggle

Le notebook `pokemon-card-grade.ipynb` de Kaggle entraine un modele YOLO de detection (`YOLO("yolo11n")`) sur le dataset `adriantseee2/card-grader` au format:

```text
Card Grader.v1i.yolov11/
  data.yaml
  train/images + train/labels
  valid/images + valid/labels
```

Il ne predit pas directement une note finale 8/9/10: il detecte des classes avec des bounding boxes, puis compare predictions et annotations sur des images de validation. C'est utile comme axe complementaire pour reperer des zones/defauts visibles, mais notre pipeline principal garde la classification front/back paire pour estimer la note finale.

Commandes conservees:

```powershell
python pokescan_amd.py --mode train
python pokescan_amd.py --mode eval --model weights/best.pt --samples 20 --save-dir runs/eval
python pokescan_amd.py --mode predict --model weights/best.pt --source image.jpg
```

Notes importantes:

- `train` peut telecharger le dataset Kaggle si `kagglehub` est configure.
- `eval` ne telecharge pas automatiquement le dataset: il doit deja exister dans `dataset/Card Grader.v1i.yolov11`.
- `predict` exige un modele existant via `--model` ou `weights/best.pt`.
- Si ROCm/HIP n'est pas detecte, la CLI bascule en CPU avec un diagnostic.

## Checks rapides

```powershell
python check_dataset.py
python pokescan_amd.py --help
python scrapper.py --help
pytest
```

## Diagnostic ROCm WSL

Un environnement WSL isole peut etre cree dans `.venv-wsl` pour tester PyTorch ROCm sans toucher au venv Windows.

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/Users/L3o36/pokescan && .venv-wsl/bin/python -u scripts/rocm_smoke.py"
```

Etat observe sur RX 6750 XT:

- ROCm/WSL detecte la carte avec `rocminfo` en `gfx1030` via `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
- PyTorch ROCm 7.2 detecte `AMD Radeon RX 6750 XT`.
- La premiere allocation GPU bloque actuellement avec le message `Windows driver is old, please update it`.

Tant que `scripts/rocm_smoke.py` ne finit pas par `ok`, l'entrainement YOLO doit rester en CPU ou attendre une mise a jour du driver AMD Windows compatible WSL ROCm.
