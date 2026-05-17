import sys

sys.path.insert(0, "src")

from pokescan.identify import card_db
from pokescan.identify.models import CardCandidate, OCRResult


def test_search_tcgdex_does_not_fallback_to_english_for_french(monkeypatch):
    calls = []

    def fake_search_by_name(name, language):
        calls.append((name, language))
        if language == "en":
            return [
                CardCandidate(
                    card_id="en-1",
                    name="Plusle",
                    set_name="",
                    set_id="",
                    number="1",
                    number_total=None,
                    language="en",
                )
            ]
        return []

    monkeypatch.setattr(card_db, "search_by_name", fake_search_by_name)

    result = card_db.search_tcgdex(OCRResult(name="Negapi", language="fr"), language="fr")

    assert result == []
    assert calls == [("Negapi", "fr")]


def test_search_tcgdex_retries_noisy_name_prefix_in_same_language(monkeypatch):
    calls = []
    expected = CardCandidate(
        card_id="pl2-42",
        name="Hippodocus  Niv. 52",
        set_name="",
        set_id="pl2",
        number="42",
        number_total=None,
        language="fr",
    )

    def fake_search_by_name(name, language):
        calls.append((name, language))
        return [expected] if (name, language) == ("Hippodocus", "fr") else []

    monkeypatch.setattr(card_db, "search_by_name", fake_search_by_name)

    result = card_db.search_tcgdex(OCRResult(name="Hippodocus W wi.", language="fr"), language="fr")

    assert result == [expected]
    assert all(language == "fr" for _name, language in calls)
