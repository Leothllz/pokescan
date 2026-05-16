"""Quick smoke test for the identify pipeline."""
import sys
sys.path.insert(0, "src")

from pokescan.identify.card_db import search_by_name, get_card_detail

# Test search
results = search_by_name("Pyroli V", "fr")
print(f"Search 'Pyroli V': {len(results)} results")
if results:
    r = results[0]
    print(f"  First: {r.name} ({r.card_id})")

# Test detail
detail = get_card_detail("swsh7-169", "fr")
if detail:
    print(f"  Detail: {detail['name']}")
    print(f"  HP: {detail.get('hp')}")
    cm = detail.get("pricing", {}).get("cardmarket", {})
    print(f"  CardMarket avg: {cm.get('avg')}EUR")
else:
    print("  Detail fetch failed")

print("\nAll tests passed!")
