import subprocess
import sys
import os

def main():
    # On s'assure d'utiliser le bon environnement Python
    python_exe = os.path.join("pokescan-env", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    print("="*60)
    print("          POKESCAN TEST - VERSION RAPIDE (20 IMAGES)")
    print("="*60)
    print("\nCe script va verifier :")
    print("1. La collecte des liens sur GCC")
    print("2. La visite multi-threadee des annonces")
    print("3. Le telechargement HD Recto/Verso")
    print("4. Le tri dans les dossiers [Grade]/front et [Grade]/back\n")

    cmd = [python_exe, "scrapper.py", "--sources", "gcc", "--max", "20"]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[!] Test interrompu par l'utilisateur.")
    except Exception as e:
        print(f"\n[!] Erreur lors du test : {e}")

if __name__ == "__main__":
    main()
