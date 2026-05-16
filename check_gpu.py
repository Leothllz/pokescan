#!/usr/bin/env python3
"""
🔍 Check GPU — Vérifie la détection du GPU AMD pour PyTorch
============================================================
Ce script teste si PyTorch peut utiliser le GPU AMD RX 6750 XT.

IMPORTANT : Le spoofing gfx1031→gfx1030 fonctionne sous Linux/WSL2
            mais PAS sous Windows natif (limitation driver).
            Voir setup_amd.md → Option B pour la solution WSL2.
"""

import os
import sys
import platform

# Fix encodage Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Configuration pour AMD RX 6750 XT
# DOIT être défini AVANT l'import de torch
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["HCC_AMDGPU_TARGET"] = "gfx1030"
os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")

# IMPORTANT : NE PAS mettre CUDA_VISIBLE_DEVICES="" !
# Sous ROCm, torch.cuda = backend HIP. Le masquer bloque tout.
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import torch

print("-" * 60)
print(f"Python version  : {sys.version.split()[0]}")
print(f"Plateforme      : {platform.system()} ({platform.machine()})")
print(f"PyTorch version : {torch.__version__}")

# Vérifier si c'est une version ROCm
is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
if is_rocm:
    print(f"ROCm/HIP version: {torch.version.hip}")
else:
    print("Backend         : CPU uniquement (pas de ROCm/HIP)")

# Variables d'environnement
print(f"\nVariables d'environnement :")
for var in ["HSA_OVERRIDE_GFX_VERSION", "HCC_AMDGPU_TARGET", 
            "HSA_ENABLE_DXG_DETECTION", "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES"]:
    val = os.environ.get(var, "(non defini)")
    print(f"  {var} = {val}")

# Test device count
print()
try:
    device_count = torch.cuda.device_count()
    print(f"Device count    : {device_count}")
except Exception as e:
    print(f"Device count    : ERREUR - {e}")
    device_count = 0

print(f"GPU disponible  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Nom du GPU      : {gpu_name}")
    try:
        props = torch.cuda.get_device_properties(0)
        print(f"Memoire totale  : {props.total_mem / 1e9:.2f} GB")
        print(f"GCN Arch        : {props.gcnArchName if hasattr(props, 'gcnArchName') else 'N/A'}")
    except Exception as e:
        print(f"Proprietes      : Erreur - {e}")
    
    # Test de calcul
    try:
        print("\nTest de calcul sur GPU...")
        x = torch.randn(1000, 1000).to("cuda")
        y = torch.matmul(x, x)
        z = y.sum().item()
        print(f"  MatMul 1000x1000 : OK (somme={z:.2f})")
        
        # Test de mémoire
        mem_alloc = torch.cuda.memory_allocated(0) / 1e6
        mem_reserved = torch.cuda.memory_reserved(0) / 1e6
        print(f"  Memoire allouee  : {mem_alloc:.1f} MB")
        print(f"  Memoire reservee : {mem_reserved:.1f} MB")
        
        del x, y
        torch.cuda.empty_cache()
        print("  GPU OK !")
    except Exception as e:
        print(f"  Test echoue : {e}")
else:
    print("\n--- GPU non detecte ---")
    
    if is_rocm and platform.system() == "Windows":
        print("""
DIAGNOSTIC : Vous etes sur Windows avec PyTorch ROCm.
Le spoofing gfx1031->gfx1030 NE FONCTIONNE PAS sur Windows natif
pour les cartes RDNA2 (RX 6700/6750 XT).

SOLUTION : Utilisez WSL2 (Ubuntu) ou le spoofing fonctionne.

  Etapes :
  1. PowerShell (admin) : wsl --install -d Ubuntu
  2. Dans WSL2 :
     cd /mnt/c/Users/VOTRE_USER/pokescan
     chmod +x setup_wsl2_rocm.sh
     ./setup_wsl2_rocm.sh

  Voir setup_amd.md pour les details complets.
""")
    elif is_rocm:
        print("""
Conseils pour ROCm sur Linux/WSL2 :
  1. Verifiez que ROCm est installe : rocminfo
  2. HSA_OVERRIDE_GFX_VERSION=10.3.0 doit etre defini
  3. HSA_ENABLE_DXG_DETECTION=1 (pour WSL2)
  4. Votre utilisateur doit etre dans les groupes video et render
     sudo usermod -aG video,render $USER
  5. Redemarrez WSL2 apres l'installation.
""")
    else:
        print("""
PyTorch n'est pas compile avec ROCm/HIP.
Installez la version ROCm :
  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
""")

print("-" * 60)
