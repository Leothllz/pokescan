#!/usr/bin/env python3
"""
🔧 Diagnostic GPU avancé — AMD RX 6750 XT (gfx1031) sur Windows
================================================================

Ce script teste TOUTES les méthodes pour faire fonctionner ta carte avec PyTorch :
  1. ROCm/HIP natif (avec spoofing gfx1030 et gfx1100)
  2. DirectML (Microsoft, compatible avec toutes les cartes DX12)
  3. CPU (fallback)

Usage : python gpu_diagnostic.py
"""

import os
import sys

# Fix encodage Windows (cp1252 ne supporte pas les caractères Unicode étendus)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import subprocess
import importlib
import time


def section(title):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def test_rocm_hip():
    """Test ROCm/HIP avec différentes versions de spoofing."""
    section("🔴 TEST 1 : ROCm/HIP (Natif Windows)")
    
    # Variantes de spoofing à tester
    spoof_configs = [
        ("10.3.0", "gfx1030", "RDNA2 - RX 6700 XT compatible"),
        ("11.0.0", "gfx1100", "RDNA3 - RX 7900 XTX (agressif)"),
        ("10.3.1", "gfx1031", "Identité réelle (gfx1031)"),
    ]
    
    for gfx_ver, target, desc in spoof_configs:
        print(f"\n  ▶ Tentative: HSA={gfx_ver}, TARGET={target} ({desc})")
        
        # Set env vars BEFORE importing torch
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = gfx_ver
        os.environ["HCC_AMDGPU_TARGET"] = target
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # Ne pas masquer le GPU
        
        # On doit tester dans un sous-processus car torch est déjà importé
        test_script = f'''
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "{gfx_ver}"
os.environ["HCC_AMDGPU_TARGET"] = "{target}"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["AMD_LOG_LEVEL"] = "0"
# NE PAS mettre CUDA_VISIBLE_DEVICES="" pour ROCm
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

try:
    import torch
    hip_ver = getattr(torch.version, "hip", None)
    if hip_ver is None:
        print("NOT_ROCM")
    elif torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        # Quick compute test
        x = torch.randn(512, 512, device="cuda")
        y = torch.matmul(x, x)
        del x, y
        torch.cuda.empty_cache()
        print(f"SUCCESS|{{name}}|{{mem:.1f}}|{{hip_ver}}")
    else:
        print("NO_DEVICE")
except Exception as e:
    print(f"ERROR|{{str(e)[:200]}}")
'''
        
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True, text=True, timeout=30,
                env={**os.environ}
            )
            output = (result.stdout.strip() + result.stderr.strip()).strip()
            
            if "SUCCESS" in output:
                parts = output.split("SUCCESS|")[1].split("|")
                gpu_name = parts[0]
                gpu_mem = parts[1]
                hip_ver = parts[2]
                print(f"    ✅ SUCCÈS ! GPU={gpu_name}, VRAM={gpu_mem} Go, HIP={hip_ver}")
                return {
                    "method": "rocm",
                    "gfx_version": gfx_ver,
                    "target": target,
                    "gpu_name": gpu_name,
                    "vram": gpu_mem,
                }
            elif "NOT_ROCM" in output:
                print("    ❌ PyTorch n'est pas compilé avec ROCm/HIP.")
                break  # Inutile de tester d'autres configs
            elif "NO_DEVICE" in output:
                print("    ❌ Pas de GPU détecté avec cette config.")
            else:
                error_msg = output.replace("\n", " ")[:150]
                print(f"    ❌ Erreur : {error_msg}")
        except subprocess.TimeoutExpired:
            print("    ❌ Timeout (le driver a peut-être planté).")
        except Exception as e:
            print(f"    ❌ Exception : {e}")
    
    return None


def test_directml():
    """Test DirectML (Microsoft) — compatible avec toutes les cartes DX12."""
    section("🟡 TEST 2 : DirectML (Microsoft)")
    
    try:
        import torch_directml
        dml_device = torch_directml.device()
        device_name = torch_directml.device_name(0)
        print(f"  ✅ DirectML disponible !")
        print(f"     Device : {device_name}")
        print(f"     Device object : {dml_device}")
        
        # Test de calcul
        import torch
        x = torch.randn(512, 512).to(dml_device)
        y = torch.matmul(x, x)
        result = y.cpu()
        del x, y
        print(f"  ✅ Test de calcul MatMul réussi !")
        
        return {
            "method": "directml",
            "device": str(dml_device),
            "device_name": device_name,
        }
    except ImportError:
        print("  ⚠️  torch-directml n'est pas installé.")
        print("     Pour l'installer :")
        print("     pip install torch-directml")
        return None
    except Exception as e:
        print(f"  ❌ Erreur DirectML : {e}")
        return None


def test_cpu():
    """Test CPU — toujours disponible."""
    section("🟢 TEST 3 : CPU (Fallback)")
    
    import torch
    print(f"  PyTorch : {torch.__version__}")
    
    # Test de performance
    start = time.perf_counter()
    x = torch.randn(1000, 1000)
    y = torch.matmul(x, x)
    elapsed = time.perf_counter() - start
    
    print(f"  ✅ CPU fonctionnel")
    print(f"     MatMul 1000x1000 en {elapsed*1000:.1f} ms")
    print(f"     Threads : {torch.get_num_threads()}")
    
    return {
        "method": "cpu",
        "perf_ms": f"{elapsed*1000:.1f}",
        "threads": torch.get_num_threads(),
    }


def check_driver_info():
    """Vérifie les infos du driver AMD."""
    section("📋 INFOS SYSTÈME")
    
    print(f"  Python : {sys.version}")
    print(f"  OS : {sys.platform}")
    
    # Vérifier si le driver AMD PyTorch Edition est installé
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "Name,DriverVersion"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip() and 'Name' not in l]
            for line in lines:
                print(f"  GPU trouvé : {line}")
    except Exception:
        pass
    
    # Vérifier les DLLs HIP
    hip_paths = [
        r"C:\Windows\System32\amdhip64.dll",
        r"C:\Program Files\AMD\ROCm",
    ]
    for path in hip_paths:
        exists = os.path.exists(path)
        print(f"  {'✅' if exists else '❌'} {path} {'(trouvé)' if exists else '(manquant)'}")
    
    # Variables d'environnement pertinentes
    env_vars = ["HSA_OVERRIDE_GFX_VERSION", "HCC_AMDGPU_TARGET", 
                "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "ROCM_HOME", "HIP_PATH"]
    print("\n  Variables d'environnement :")
    for var in env_vars:
        val = os.environ.get(var, "(non défini)")
        print(f"    {var} = {val}")


def generate_fix_script(best_result):
    """Génère les instructions de fix basées sur le meilleur résultat."""
    section("🛠️ RECOMMANDATION")
    
    if best_result is None:
        print("""
  ❌ Aucune méthode GPU n'a fonctionné.
  
  Options restantes :
  
  1. INSTALLER DirectML (meilleure option Windows) :
     pip install torch-directml
     Puis relancer ce diagnostic.
  
  2. WSL2 + ROCm (méthode la plus fiable pour RDNA2) :
     Le spoofing gfx1030 fonctionne BEAUCOUP mieux sous Linux.
     Voir setup_amd.md section "Option B".
  
  3. Vérifier le driver :
     Installer "AMD Software: PyTorch on Windows Edition"
     (pas le driver Adrenalin standard !)
     https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
""")
        return
    
    method = best_result["method"]
    
    if method == "rocm":
        print(f"""
  ✅ ROCm/HIP fonctionne avec le spoofing !
  
  Configuration qui marche :
    HSA_OVERRIDE_GFX_VERSION = {best_result['gfx_version']}
    HCC_AMDGPU_TARGET = {best_result['target']}
  
  GPU : {best_result['gpu_name']} ({best_result['vram']} Go VRAM)
  
  ⚠️  IMPORTANT : NE PAS mettre CUDA_VISIBLE_DEVICES=""
      Cela empêche ROCm de voir le GPU !
  
  Le script pokescan_amd.py va être mis à jour automatiquement.
""")
    
    elif method == "directml":
        print(f"""
  ✅ DirectML fonctionne !
  
  Device : {best_result['device_name']}
  
  Pour l'utiliser avec YOLO/Ultralytics :
    model.train(data="data.yaml", device="dml", ...)
  
  Note : DirectML est plus stable que ROCm sur Windows
  pour les cartes RDNA2 non supportées officiellement.
  
  Le script pokescan_amd.py va être mis à jour automatiquement.
""")
    
    elif method == "cpu":
        print(f"""
  ℹ️  Seul le CPU est disponible.
  
  Performance : MatMul 1000x1000 en {best_result['perf_ms']} ms
  Threads : {best_result['threads']}
  
  YOLO26 est optimisé pour le CPU (jusqu'à 43% de gain).
  L'entraînement sera plus lent mais fonctionnel.
""")


def main():
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  🔧 Diagnostic GPU — AMD RX 6750 XT (gfx1031) sur Windows  ║")
    print("║  Test de toutes les méthodes pour PyTorch                   ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    check_driver_info()
    
    best = None
    
    # Test 1 : ROCm/HIP
    result = test_rocm_hip()
    if result:
        best = result
    
    # Test 2 : DirectML
    if best is None:
        result = test_directml()
        if result:
            best = result
    
    # Test 3 : CPU (toujours disponible)
    if best is None:
        best = test_cpu()
    
    generate_fix_script(best)
    
    return best


if __name__ == "__main__":
    result = main()
