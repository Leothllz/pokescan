#!/usr/bin/env python3
"""PokeScan Card Identifier — CLI.

Usage:
    python identify_card.py --source photo.jpg
    python identify_card.py --source photo.jpg --lang fr --top-k 3 --verbose
    python identify_card.py --source folder/ --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def _print_result(result, verbose: bool = False) -> None:
    from pokescan.identify.models import CardIdentity

    r: CardIdentity = result

    print()
    print(f"{Colors.BOLD}{'═' * 52}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}  IDENTIFICATION DE CARTE{Colors.END}")
    print(f"{Colors.BOLD}{'═' * 52}{Colors.END}")

    # OCR results.
    ocr = r.ocr_result
    print()
    print(f"{Colors.CYAN}  OCR détecté:{Colors.END}")
    print(f"    Nom      : {Colors.BOLD}{ocr.name or '—'}{Colors.END}")
    print(f"    Numéro   : {ocr.collector_number or '—'}")
    print(f"    HP       : {ocr.hp or '—'}")
    print(f"    Langue   : {(ocr.language or '—').upper()}")
    print(f"    Année    : {ocr.year or '—'}")

    if verbose and ocr.raw_texts:
        print(f"\n{Colors.DIM}  Textes bruts:{Colors.END}")
        for zone, text in ocr.raw_texts.items():
            print(f"    {zone}: {Colors.DIM}{text[:80]}{Colors.END}")

    if r.best_match:
        m = r.best_match
        conf_color = Colors.GREEN if r.confidence > 0.7 else Colors.YELLOW if r.confidence > 0.4 else Colors.RED
        number_display = f"{m.number}/{m.number_total}" if m.number_total else m.number

        print()
        print(f"  {conf_color}Meilleur match (confiance: {r.confidence:.1%}):{Colors.END}")
        print(f"    Nom      : {Colors.BOLD}{m.name}{Colors.END}")
        print(f"    Extension: {m.set_name} ({m.set_id})")
        print(f"    Numéro   : {number_display}")
        print(f"    Rareté   : {m.rarity or '—'}")
        if m.hp:
            print(f"    HP       : {m.hp}")
        if m.image_url:
            print(f"    Image    : {Colors.DIM}{m.image_url}{Colors.END}")

        # Pricing.
        if m.pricing:
            print()
            print(f"  {Colors.CYAN}Prix (source: TCGdex):{Colors.END}")
            cm = m.pricing.get("cardmarket")
            if cm:
                avg = cm.get("avg")
                trend = cm.get("trend")
                low = cm.get("low")
                print(f"    CardMarket : {Colors.GREEN}~{avg}€{Colors.END} (trend: {trend}€, low: {low}€)")
            tp = m.pricing.get("tcgplayer")
            if tp and isinstance(tp, dict):
                for variant, data in tp.items():
                    if isinstance(data, dict) and "marketPrice" in data:
                        print(f"    TCGPlayer ({variant}): ${data['marketPrice']}")
            if not cm and not tp:
                print(f"    {Colors.DIM}Pas de prix disponible{Colors.END}")

        if verbose:
            print(f"\n{Colors.DIM}  Score détail: {m.score_detail}{Colors.END}")
    else:
        print()
        print(f"  {Colors.RED}Aucun match trouvé.{Colors.END}")
        print(f"  {Colors.YELLOW}Conseils :{Colors.END}")
        print(f"    - Vérifiez que la photo est nette et bien cadrée")
        print(f"    - Essayez --lang en si la carte est en anglais")

    # Other candidates.
    if verbose and len(r.candidates) > 1:
        print(f"\n  {Colors.CYAN}Autres candidats:{Colors.END}")
        for i, c in enumerate(r.candidates[1:], 2):
            num = f"{c.number}/{c.number_total}" if c.number_total else c.number
            print(f"    #{i} {c.name} ({c.set_name}) {num} — {c.score:.1%}")

    print(f"\n{Colors.BOLD}{'═' * 52}{Colors.END}")


def _result_to_dict(result) -> dict:
    """Convert a CardIdentity to a JSON-serializable dict."""
    r = result
    d: dict = {
        "ocr": {
            "name": r.ocr_result.name,
            "collector_number": r.ocr_result.collector_number,
            "hp": r.ocr_result.hp,
            "language": r.ocr_result.language,
            "year": r.ocr_result.year,
        },
        "confidence": r.confidence,
        "best_match": None,
        "candidates": [],
    }
    if r.best_match:
        m = r.best_match
        d["best_match"] = {
            "card_id": m.card_id,
            "name": m.name,
            "set_name": m.set_name,
            "set_id": m.set_id,
            "number": m.number,
            "number_total": m.number_total,
            "rarity": m.rarity,
            "hp": m.hp,
            "image_url": m.image_url,
            "pricing": m.pricing,
            "score": m.score,
        }
    for c in r.candidates:
        d["candidates"].append({
            "card_id": c.card_id,
            "name": c.name,
            "set_name": c.set_name,
            "number": c.number,
            "score": c.score,
        })
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PokeScan — Identification de cartes Pokémon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Image ou dossier à identifier")
    parser.add_argument("--lang", default="fr", choices=["fr", "en"], help="Langue principale")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de candidats")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", default=None, help="Fichier JSON de sortie")
    parser.add_argument("--skip-crop", action="store_true", help="Image déjà croppée")

    args = parser.parse_args(argv)

    from pokescan.identify.pipeline import identify_card

    source = Path(args.source)
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(f for f in source.iterdir() if f.suffix.lower() in image_exts)
    else:
        print(f"Source introuvable: {source}")
        return 1

    if not files:
        print(f"Aucune image trouvée dans {source}")
        return 1

    all_results = []
    for img_path in files:
        print(f"\n{'─' * 40}")
        print(f"  Analyse: {img_path.name}")
        print(f"{'─' * 40}")

        result = identify_card(
            img_path,
            language=args.lang,
            top_k=args.top_k,
            skip_crop=args.skip_crop,
        )
        _print_result(result, verbose=args.verbose)
        all_results.append({"file": str(img_path), **_result_to_dict(result)})

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nRésultats sauvegardés dans {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
