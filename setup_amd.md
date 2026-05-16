# 🛠️ Guide d'installation — PokeScan sur AMD RX 6750 XT (Windows & Linux)

## ⚠️ Avertissement important

La RX 6750 XT (GPU Navi 22, architecture `gfx1031`) **n'est pas officiellement supportée** par ROCm.
On utilise un « spoof » pour la faire passer pour une `gfx1030` (RDNA 2).

> **⚡ IMPORTANT :** Le spoofing fonctionne uniquement sous **Linux/WSL2**.  
> Il **ne fonctionne PAS** sous Windows natif (le driver filtre les GPU au niveau kernel).  
> **→ L'Option B (WSL2) est RECOMMANDÉE pour Windows.**

---

## ~~Option A : Installation Native Windows~~ ❌ (Non fonctionnel pour RX 6750 XT)

> **Ne marche PAS pour les cartes RDNA2 (RX 6700/6750 XT).**  
> L'option native Windows ne fonctionne que pour les séries RX 7000/9000.  
> Le driver Windows bloque les GPU gfx1031 au niveau kernel, même avec le spoofing.  
> 
> Si vous avez cette option installée, elle fonctionnera en mode CPU uniquement.

---

## Option B : WSL2 sur Windows ⭐ (Recommandé !)

Le spoofing `gfx1031 → gfx1030` fonctionne **parfaitement sous Linux/WSL2**.
C'est la méthode la plus fiable pour utiliser ta RX 6750 XT avec PyTorch.

### Installation automatique

Un script d'installation est fourni. Depuis PowerShell :

```powershell
# 1. Installer WSL2 (si pas déjà fait)
wsl --install -d Ubuntu

# 2. Redémarrer le PC si demandé, puis ouvrir Ubuntu

# 3. Dans le terminal Ubuntu/WSL2 :
cd /mnt/c/Users/VOTRE_USER/pokescan
chmod +x setup_wsl2_rocm.sh
./setup_wsl2_rocm.sh
```

### Installation manuelle WSL2

#### 1. Installer ROCm dans WSL2
```bash
# Clé GPG ROCm
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg

# Dépôt ROCm 6.2
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2/ jammy main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update && sudo apt install -y rocm-dev rocm-libs rocm-hip-sdk
```

#### 2. Installer librocdxg (pont WSL2 ↔ GPU Windows)
```bash
git clone https://github.com/ROCm/librocdxg.git ~/librocdxg
cd ~/librocdxg
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/rocm
make -j$(nproc)
sudo make install
```

#### 3. Variables d'environnement
Ajoutez dans `~/.bashrc` :
```bash
# ROCm
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

# Spoofing RX 6750 XT (gfx1031 → gfx1030)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HCC_AMDGPU_TARGET=gfx1030

# Activer la détection DXG pour WSL2
export HSA_ENABLE_DXG_DETECTION=1
```

#### 4. PyTorch ROCm & Dépendances
```bash
python3 -m venv ~/pokescan-env
source ~/pokescan-env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install ultralytics kagglehub opencv-python-headless matplotlib
```

#### 5. Vérification
```bash
# Vérifier que le GPU est détecté
source ~/pokescan-env/bin/activate
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

#### 6. Groupes utilisateur (important !)
```bash
sudo usermod -aG video,render $USER
# Puis redémarrer WSL2 : wsl --shutdown (depuis PowerShell)
```

---

## Option C : Linux Natif (Dual-Boot)

Si WSL2 pose problème, un dual-boot Linux est aussi une option.

```bash
# Suivre les mêmes étapes que WSL2, sauf :
# - Pas besoin de librocdxg
# - Installer le driver amdgpu natif :
sudo apt install -y amdgpu-dkms
# - Le reste est identique (ROCm + spoofing)
```

---

## Option D : CPU uniquement (Fallback)

Si ROCm n'est pas disponible, YOLO26 est **optimisé pour le CPU** avec jusqu'à 43% de gain en inférence.
```cmd
pip install torch torchvision ultralytics kagglehub opencv-python matplotlib
python pokescan_amd.py --mode train --cpu --epochs 50 --batch 8
```

---

## 🚀 Utilisation

### Depuis WSL2 (recommandé)
```bash
# Activer l'environnement
source ~/pokescan-env/bin/activate

# Aller dans le dossier (accessible via /mnt/c/)
cd /mnt/c/Users/VOTRE_USER/pokescan

# Entraîner
python3 pokescan_amd.py --mode train --model-size n --epochs 100 --batch 16

# Évaluer
python3 pokescan_amd.py --mode eval --samples 15

# Prédire
python3 pokescan_amd.py --mode predict --model weights/best.pt --source ma_carte.jpg
```

### Depuis Windows (CPU uniquement)
```cmd
pokescan-env\Scripts\activate
python pokescan_amd.py --mode train --cpu --epochs 50 --batch 8
```

---

## 📊 Tailles de batch recommandées (RX 6750 XT — 12 Go VRAM)

| Modèle    | Batch max (640px) |
|-----------|-------------------|
| yolo26n   | 32                |
| yolo26s   | 16                |
| yolo26m   | 8                 |
| yolo26l   | 4                 |

---

## 🐛 Dépannage

| Problème | Solution |
|----------|----------|
| `Failed to get device count` (Windows) | Le spoofing ne marche pas sur Windows natif → Utiliser WSL2 |
| `Failed to get device count` (WSL2) | Vérifier `HSA_ENABLE_DXG_DETECTION=1` et le driver Windows AMD |
| GPU non détecté dans WSL2 | `sudo usermod -aG video,render $USER` puis `wsl --shutdown` |
| `rocminfo` ne trouve rien | Driver AMD Windows trop ancien → Mettre à jour Adrenalin |
| OOM (Out of Memory) | Réduire le batch size ou la résolution (`--imgsz 416`) |
| `CUDA_VISIBLE_DEVICES` bloque ROCm | NE PAS définir cette variable, elle masque aussi HIP |
