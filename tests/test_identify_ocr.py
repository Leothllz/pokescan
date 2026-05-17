import sys

import numpy as np

sys.path.insert(0, "src")

from pokescan.identify import ocr


def test_extract_card_text_cleans_name_and_footer_noise(monkeypatch):
    monkeypatch.setenv("POKESCAN_OCR_SIDECAR_ACTIVE", "1")
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    def fake_read_zone(_image, zone):
        if zone == ocr.ZONE_NAME:
            return "Onix 90 HP Baslc Pokemon"
        if zone == ocr.ZONE_HP:
            return "Oo UD @"
        if zone == ocr.ZONE_BOTTOM:
            return "LY 40 #9S ~ Illus 91595-2091 Nintendo"
        if zone == ocr.ZONE_NUMBER:
            return "3/18"
        return ""

    monkeypatch.setattr(ocr, "_read_zone", fake_read_zone)

    result = ocr.extract_card_text(image)

    assert result.name == "Onix"
    assert result.hp == "90"
    assert result.collector_number == "3/18"
    assert result.language == "en"
    assert result.year is None


def test_extract_card_text_keeps_plausible_latest_year(monkeypatch):
    monkeypatch.setenv("POKESCAN_OCR_SIDECAR_ACTIVE", "1")
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    def fake_read_zone(_image, zone):
        if zone == ocr.ZONE_NAME:
            return "Pikachu PV 70 Pokemon de base"
        if zone == ocr.ZONE_HP:
            return ""
        if zone == ocr.ZONE_BOTTOM:
            return "25/198 ©1995-2023 Nintendo"
        if zone == ocr.ZONE_NUMBER:
            return "25/198"
        return ""

    monkeypatch.setattr(ocr, "_read_zone", fake_read_zone)

    result = ocr.extract_card_text(image)

    assert result.name == "Pikachu"
    assert result.hp == "70"
    assert result.collector_number == "25/198"
    assert result.language == "fr"
    assert result.year == "2023"


def test_extract_card_text_cleans_french_pv_noise(monkeypatch):
    monkeypatch.setenv("POKESCAN_OCR_SIDECAR_ACTIVE", "1")
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    def fake_read_zone(_image, zone):
        if zone == ocr.ZONE_NAME:
            return "Negapi, = 60 P 7,9 ( € ,"
        if zone == ocr.ZONE_HP:
            return "605"
        return ""

    monkeypatch.setattr(ocr, "_read_zone", fake_read_zone)

    result = ocr.extract_card_text(image)

    assert result.name == "Negapi"
    assert result.hp == "60"
    assert result.language is None


def test_extract_card_text_prefers_pv_over_level(monkeypatch):
    monkeypatch.setenv("POKESCAN_OCR_SIDECAR_ACTIVE", "1")
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    def fake_read_zone(_image, zone):
        if zone == ocr.ZONE_NAME:
            return "Hippodocus MIY.52 Pv 90 Base HIPpodocus % n.52 BASE"
        if zone == ocr.ZONE_HP:
            return "Pv 90"
        if zone == ocr.ZONE_BODY_TEXT:
            return "faiblesse Resistance Armure de sable"
        return ""

    def fake_read_zone_raw(_image, zone):
        return fake_read_zone(_image, zone)

    monkeypatch.setattr(ocr, "_read_zone", fake_read_zone)
    monkeypatch.setattr(ocr, "_read_zone_raw", fake_read_zone_raw)

    result = ocr.extract_card_text(image)

    assert result.name == "Hippodocus"
    assert result.hp == "90"
    assert result.language == "fr"


def test_extract_card_text_detects_french_pv_without_space(monkeypatch):
    monkeypatch.setenv("POKESCAN_OCR_SIDECAR_ACTIVE", "1")
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    def fake_read_zone(_image, zone):
        if zone == ocr.ZONE_NAME:
            return "Negapi 60PV"
        if zone == ocr.ZONE_HP:
            return "60PV"
        return ""

    monkeypatch.setattr(ocr, "_read_zone", fake_read_zone)
    monkeypatch.setattr(ocr, "_read_zone_raw", fake_read_zone)

    result = ocr.extract_card_text(image)

    assert result.name == "Negapi"
    assert result.hp == "60"
    assert result.language == "fr"


def test_extract_card_text_cleans_level_noise_and_reads_footer_number(monkeypatch):
    monkeypatch.setenv("POKESCAN_OCR_SIDECAR_ACTIVE", "1")
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    def fake_read_zone(_image, zone):
        if zone == ocr.ZONE_NAME:
            return "Hippodocus W wi.32 PV 90 BASE"
        if zone == ocr.ZONE_HP:
            return "PV 90"
        return ""

    def fake_read_zone_raw(_image, zone):
        if zone == ocr.ZONE_FOOTER:
            return "©2009 Pokémon/Nintendo 42/111 <"
        return fake_read_zone(_image, zone)

    monkeypatch.setattr(ocr, "_read_zone", fake_read_zone)
    monkeypatch.setattr(ocr, "_read_zone_raw", fake_read_zone_raw)

    result = ocr.extract_card_text(image)

    assert result.name == "Hippodocus"
    assert result.collector_number == "42/111"
    assert result.hp == "90"
    assert result.language == "fr"
    assert result.year == "2009"
