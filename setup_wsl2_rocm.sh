#!/bin/bash
# =============================================================================
# 🔧 Setup ROCm dans WSL2 pour AMD RX 6750 XT (gfx1031)
# =============================================================================
# Ce script installe et configure ROCm dans WSL2 (Ubuntu) pour faire
# fonctionner PyTorch avec ta RX 6750 XT via le spoofing gfx1030.
#
# Prérequis (côté Windows) :
#   1. Windows 11 avec WSL2 activé
#   2. Ubuntu 22.04 ou 24.04 installé dans WSL2
#   3. Driver AMD Adrenalin Edition (dernière version) sur Windows
#
# Usage :
#   chmod +x setup_wsl2_rocm.sh
#   ./setup_wsl2_rocm.sh
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
}

print_ok() {
    echo -e "${GREEN}  ✅ $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}  ⚠️  $1${NC}"
}

print_err() {
    echo -e "${RED}  ❌ $1${NC}"
}

# =============================================================================
# Étape 0 : Vérification de l'environnement
# =============================================================================
print_step "Étape 0 : Vérification de l'environnement"

if ! grep -qi microsoft /proc/version 2>/dev/null; then
    print_err "Ce script doit être exécuté dans WSL2 (Ubuntu), pas sur Windows natif !"
    exit 1
fi
print_ok "Environnement WSL2 détecté"

# Vérifier Ubuntu version
. /etc/os-release
echo "  Distribution : $NAME $VERSION_ID"

# =============================================================================
# Étape 1 : Mise à jour du système
# =============================================================================
print_step "Étape 1 : Mise à jour du système"

sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3 python3-pip python3-venv \
    build-essential cmake git wget curl \
    libfmt-dev libdrm-dev libelf-dev \
    mesa-common-dev

print_ok "Système mis à jour"

# =============================================================================
# Étape 2 : Installation de ROCm
# =============================================================================
print_step "Étape 2 : Installation de ROCm"

# Ajouter le dépôt ROCm
if [ ! -f /etc/apt/sources.list.d/rocm.list ]; then
    echo "  Ajout du dépôt ROCm..."
    
    # Clé GPG
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
    
    # Dépôt ROCm 6.2 (dernière version stable compatible RDNA2 sous Linux/WSL2)
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2/ jammy main" \
        | sudo tee /etc/apt/sources.list.d/rocm.list
    
    # Priorité
    echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
        | sudo tee /etc/apt/preferences.d/rocm-pin-600
    
    sudo apt update
    print_ok "Dépôt ROCm ajouté"
else
    print_ok "Dépôt ROCm déjà configuré"
fi

# Installer les paquets ROCm
sudo apt install -y rocm-dev rocm-libs rocm-hip-sdk hipcc

print_ok "ROCm installé"

# =============================================================================
# Étape 3 : Installation de librocdxg (pont WSL2 ↔ GPU Windows)
# =============================================================================
print_step "Étape 3 : Installation de librocdxg"

ROCDXG_DIR="$HOME/librocdxg"
if [ ! -d "$ROCDXG_DIR" ]; then
    echo "  Clonage de librocdxg..."
    git clone https://github.com/ROCm/librocdxg.git "$ROCDXG_DIR"
fi

cd "$ROCDXG_DIR"
git pull --quiet

# Build et installation
if [ ! -f /opt/rocm/lib/librocdxg.so ]; then
    echo "  Compilation de librocdxg..."
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/rocm
    make -j$(nproc)
    sudo make install
    print_ok "librocdxg compilé et installé"
else
    print_ok "librocdxg déjà installé"
fi

# =============================================================================
# Étape 4 : Configuration des variables d'environnement
# =============================================================================
print_step "Étape 4 : Configuration des variables d'environnement"

BASHRC="$HOME/.bashrc"
MARKER="# === PokeScan ROCm Configuration ==="

if ! grep -q "$MARKER" "$BASHRC"; then
    cat >> "$BASHRC" << 'EOF'

# === PokeScan ROCm Configuration ===
# ROCm paths
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

# Spoofing RX 6750 XT (gfx1031 → gfx1030)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HCC_AMDGPU_TARGET=gfx1030

# Activer la détection DXG pour WSL2
export HSA_ENABLE_DXG_DETECTION=1

# Réduire les messages de log AMD
export AMD_LOG_LEVEL=0
# === Fin PokeScan ROCm Configuration ===
EOF
    print_ok "Variables ajoutées à ~/.bashrc"
else
    print_ok "Variables déjà configurées dans ~/.bashrc"
fi

# Appliquer immédiatement
source "$BASHRC"

# Ajouter l'utilisateur aux groupes nécessaires
sudo usermod -aG video,render "$USER" 2>/dev/null || true
print_ok "Utilisateur ajouté aux groupes video/render"

# =============================================================================
# Étape 5 : Vérification ROCm
# =============================================================================
print_step "Étape 5 : Vérification ROCm"

# Tester rocminfo
if command -v rocminfo &>/dev/null; then
    echo "  Exécution de rocminfo..."
    GPU_INFO=$(rocminfo 2>/dev/null | grep -i "name:" | head -5 || echo "")
    if [ -n "$GPU_INFO" ]; then
        echo "$GPU_INFO" | while read -r line; do
            echo "    $line"
        done
        print_ok "rocminfo fonctionne"
    else
        print_warn "rocminfo n'a pas trouvé de GPU. Vérifiez vos drivers Windows."
    fi
else
    print_warn "rocminfo non trouvé. ROCm peut ne pas être correctement installé."
fi

# =============================================================================
# Étape 6 : Environnement Python + PyTorch ROCm
# =============================================================================
print_step "Étape 6 : Environnement Python + PyTorch ROCm"

VENV_DIR="$HOME/pokescan-env"
if [ ! -d "$VENV_DIR" ]; then
    echo "  Création de l'environnement virtuel..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
print_ok "Environnement virtuel activé : $VENV_DIR"

# Installer PyTorch ROCm
echo "  Installation de PyTorch ROCm..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Installer les dépendances du projet
echo "  Installation des dépendances PokeScan..."
pip install ultralytics --upgrade
pip install kagglehub opencv-python-headless matplotlib

print_ok "Tous les packages installés"

# =============================================================================
# Étape 7 : Test final PyTorch + GPU
# =============================================================================
print_step "Étape 7 : Test final PyTorch + GPU"

python3 -c "
import torch
print(f'  PyTorch version : {torch.__version__}')

hip_ver = getattr(torch.version, 'hip', None)
if hip_ver:
    print(f'  HIP version     : {hip_ver}')
else:
    print('  ⚠️  PyTorch n est pas compilé avec ROCm/HIP')

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f'  ✅ GPU détecté   : {name}')
    print(f'     VRAM          : {mem:.1f} Go')
    
    # Test rapide
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    print(f'  ✅ Test MatMul réussi !')
    
    del x, y
    torch.cuda.empty_cache()
else:
    print('  ❌ GPU non détecté par PyTorch')
    print('     Vérifiez :')
    print('     1. Que le driver AMD Windows est à jour')
    print('     2. Que HSA_OVERRIDE_GFX_VERSION=10.3.0 est défini')
    print('     3. Que HSA_ENABLE_DXG_DETECTION=1 est défini')
"

# =============================================================================
# Résumé
# =============================================================================
print_step "Installation terminée !"
echo -e "
  Pour utiliser PokeScan dans WSL2 :
  
  ${GREEN}1. Ouvrir un terminal WSL2${NC}
  ${GREEN}2. source ~/pokescan-env/bin/activate${NC}
  ${GREEN}3. cd /mnt/c/Users/L3o36/pokescan${NC}
  ${GREEN}4. python3 pokescan_amd.py --mode train --epochs 100${NC}
  
  Note : Tes fichiers Windows sont accessibles via /mnt/c/
"
