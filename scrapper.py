# -*- coding: utf-8 -*-
"""
PokeScan - Scrapper d'images de cartes Pokemon PSA
===================================================
Recupere des images HD de cartes Pokemon gradees PSA depuis plusieurs sources :
  1. Pokemon TCG API (gratuite, images HD officielles)
  2. eBay via Scrapy (anti-bot, user-agent rotation)
  3. Pokellector (scans de cartes)

Usage:
    python scrapper.py
    python scrapper.py --max 500 --sources all
    python scrapper.py --sources tcg --max 200
    python scrapper.py --sources ebay --max 100
"""

import requests
from bs4 import BeautifulSoup
import random
import os
import sys
import time
import json
import hashlib
import argparse
import re
from urllib.parse import quote_plus, urljoin, urlparse, urlencode
from fake_useragent import UserAgent

# ============================================================================
#                           CONFIGURATION
# ============================================================================

SAVE_FOLDER = "dataset_pokemon"
DEFAULT_MAX_IMAGES = 300

# Requetes de recherche
SEARCH_QUERIES = [
    "pokemon card psa 10",
    "pokemon card psa 9",
    "pokemon card psa graded",
    "pokemon tcg graded card",
    "pokemon psa slab",
    "pokemon card bgs 10",
    "pokemon card cgc graded",
]

# Proxies (ajouter les tiens ici si besoin)
PROXY_LIST = []

# ============================================================================
#                           UTILITAIRES
# ============================================================================

ua = UserAgent()


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def log(msg, color=Colors.END):
    try:
        print(f"{color}{msg}{Colors.END}")
    except UnicodeEncodeError:
        print(msg)


def log_success(msg):
    log(f"  [OK] {msg}", Colors.GREEN)

def log_warn(msg):
    log(f"  [!] {msg}", Colors.YELLOW)

def log_error(msg):
    log(f"  [X] {msg}", Colors.RED)

def log_info(msg):
    log(f"  > {msg}", Colors.CYAN)

def log_header(msg):
    print()
    log(f"{'='*60}", Colors.BOLD)
    log(f"  {msg}", Colors.BOLD + Colors.HEADER)
    log(f"{'='*60}", Colors.BOLD)
    print()


def get_headers():
    """Retourne des headers aleatoires pour eviter la detection."""
    return {
        "User-Agent": ua.random,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": random.choice([
            "https://www.google.com/",
            "https://www.google.fr/",
            "https://www.bing.com/",
        ]),
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }


def get_proxy():
    """Retourne un proxy aleatoire ou None."""
    if PROXY_LIST:
        p = random.choice(PROXY_LIST)
        return {"http": p, "https": p}
    return None


def get_image_hash(data):
    """Calcule un hash MD5 pour detecter les doublons."""
    return hashlib.md5(data).hexdigest()


def iter_image_files(folder):
    """Itere recursivement sur les images locales."""
    image_exts = ('.jpg', '.jpeg', '.png', '.webp')
    if not os.path.exists(folder):
        return
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(image_exts):
                yield os.path.join(root, filename)


def count_local_images(folder):
    """Compte les images locales, y compris dans grade/front et grade/back."""
    return sum(1 for _ in iter_image_files(folder))


def download_image(url, filepath, session=None, min_size_kb=5):
    """
    Telecharge une image depuis une URL.
    Retourne (True, hash) si succes, (False, None) sinon.
    """
    try:
        requester = session if session else requests
        response = requester.get(url, timeout=15, stream=True, headers=get_headers())
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        if not any(t in content_type for t in ['image/', 'octet-stream']):
            return False, None

        data = response.content

        if len(data) < min_size_kb * 1024:
            return False, None

        img_hash = get_image_hash(data)

        with open(filepath, 'wb') as f:
            f.write(data)

        return True, img_hash

    except Exception:
        return False, None


# ============================================================================
#                     SOURCE 1 : POKEMON TCG API
# ============================================================================

def scrape_pokemon_tcg_api(max_images, existing_hashes, save_folder=SAVE_FOLDER):
    """
    Utilise l'API PokemonTCG (api.pokemontcg.io) pour recuperer
    des images HD officielles de cartes.
    C'est la source la plus fiable et de meilleure qualite.
    """
    log_header("SOURCE : Pokemon TCG API")

    base_url = "https://api.pokemontcg.io/v2/cards"
    count = 0
    page = 1
    page_size = 50

    session = requests.Session()

    while count < max_images:
        log_info(f"Page API {page} (images recuperees : {count}/{max_images})")

        params = {
            "page": page,
            "pageSize": page_size,
            "orderBy": "-set.releaseDate",
        }

        try:
            response = session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            cards = data.get("data", [])
            if not cards:
                log_warn("Plus de cartes disponibles depuis l'API.")
                break

            for card in cards:
                if count >= max_images:
                    break

                images = card.get("images", {})
                img_url = images.get("large") or images.get("small")

                if not img_url:
                    continue

                card_id = card.get("id", f"unknown_{count}")
                card_name = card.get("name", "unknown").replace(" ", "_").replace("/", "-")
                set_name = card.get("set", {}).get("name", "unknown").replace(" ", "_").replace("/", "-")

                filename = f"tcg_{set_name}_{card_name}_{card_id}.png"
                filepath = os.path.join(save_folder, filename)

                if os.path.exists(filepath):
                    continue

                success, img_hash = download_image(img_url, filepath, session)

                if success:
                    if img_hash in existing_hashes:
                        os.remove(filepath)
                        continue
                    existing_hashes.add(img_hash)
                    count += 1
                    if count % 25 == 0:
                        log_success(f"{count} images telechargees...")

                time.sleep(0.1)

            page += 1
            time.sleep(1)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                log_warn("Rate limit atteint, pause de 30s...")
                time.sleep(30)
            else:
                log_error(f"Erreur API : {e}")
                break
        except Exception as e:
            log_error(f"Erreur : {e}")
            break

    log_success(f"TCG API : {count} images recuperees !")
    return count


# ============================================================================
#                     SOURCE 2 : GRADED CARD CENTER (PLAYWRIGHT)
# ============================================================================

GCC_BASE_URL = "https://gradedcardcenter.com"
GCC_LIST_URLS = [
    "https://gradedcardcenter.com/filtres/fixed-price?itemTypes=%5B%22CARDS%22%5D&categories=%5B%22Pokemon%22%5D&gradingCompanies=%5B%22PSA%22%2C%22PCA%22%5D",
    "https://gradedcardcenter.com/filtres/fixed-price?itemTypes=%5B%22CARDS%22%5D&categories=%5B%22Pokemon%22%5D&gradingCompanies=%5B%22PSA%22%2C%22PCA%22%5D&sortType=SOLD_LAST",
]
GRADE_RE = re.compile(r"\b(?:PSA|PCA|BGS|CGC|SGC)\s*(10|9\.5|9|8\.5|8|7\.5|7|6\.5|6|5|4|3|2|1)\b", re.IGNORECASE)


def extract_grade(text):
    """Extrait une note depuis un titre GCC."""
    if not text:
        return "unknown"
    match = GRADE_RE.search(text)
    return match.group(1) if match else "unknown"


def normalize_gcc_image_url(src):
    """Convertit une URL Cloudflare GCC en URL image directe haute qualite."""
    if not src or "cdn.gradedcardcenter.com" not in src or "/item" not in src:
        return None
    if src.startswith("//"):
        src = "https:" + src
    if src.startswith("/"):
        src = "https://cdn.gradedcardcenter.com" + src
    if "cdn-cgi/image" in src:
        filename = src.rsplit("/", 1)[-1]
        return f"https://cdn.gradedcardcenter.com/{filename}"
    return src


def image_side_from_url(url, fallback_index):
    lower = url.lower()
    if "verso" in lower or "back" in lower:
        return "back"
    if "recto" in lower or "front" in lower:
        return "front"
    return "front" if fallback_index == 0 else "back"


def safe_filename_part(value):
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value or "")
    return value.strip("._")[:80] or "item"


def collect_existing_hashes(save_folder):
    """Calcule les hash existants recursivement pour eviter les doublons."""
    hashes = set()
    for fpath in iter_image_files(save_folder):
        try:
            with open(fpath, "rb") as file:
                hashes.add(get_image_hash(file.read()))
        except Exception:
            pass
    return hashes


def scrape_gcc_playwright(max_images, existing_hashes, save_folder=SAVE_FOLDER):
    """
    Scrape Graded Card Center en utilisant Playwright.
    Recupere des images certifiees de cartes gradees PSA/PCA.
    """
    log_header("SOURCE : Graded Card Center (Playwright)")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log_error("Playwright n'est pas installe. Source GCC desactivee.")
        return 0

    found_items_data = []
    found_hrefs = set()
    target_items = max(10, (max_images // 2) + 10)
    
    log_info("Demarrage du navigateur Playwright...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=ua.random,
                viewport={"width": 1920, "height": 1080},
                locale="fr-FR",
            )
            page = context.new_page()

            for url in GCC_LIST_URLS:
                if len(found_hrefs) >= target_items:
                    break
                    
                log_info(f"Chargement de la liste : {url}")
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=45000)
                    page.wait_for_timeout(2500)
                    
                    try:
                        page.click('button:has-text("Accepter")', timeout=2000)
                    except:
                        pass
                    
                    log_info("Collecte des annonces GCC avec scroll...")
                    last_count = len(found_hrefs)
                    retries = 0
                    scroll_count = 0

                    while retries <= 4 and len(found_hrefs) < target_items:
                        links = page.query_selector_all('a[href*="/item/"]')
                        for link in links:
                            href = link.get_attribute("href")
                            if not href or "/item/" not in href:
                                continue

                            full_href = urljoin(GCC_BASE_URL, href)
                            if full_href in found_hrefs:
                                continue

                            title = ""
                            try:
                                img = link.query_selector('img[alt]:not([alt=""])')
                                if img:
                                    title = img.get_attribute("alt") or ""
                                if not title:
                                    title = link.inner_text().strip()
                            except Exception:
                                title = ""

                            found_hrefs.add(full_href)
                            found_items_data.append({
                                "href": full_href,
                                "title": title,
                                "grade": extract_grade(title),
                            })

                        if len(found_hrefs) >= target_items:
                            log_info(f"Quota d'annonces atteint ({len(found_hrefs)}).")
                            break

                        scroll_count += 1
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        page.wait_for_timeout(1800)

                        if len(found_hrefs) == last_count:
                            retries += 1
                        else:
                            retries = 0
                        last_count = len(found_hrefs)

                        if scroll_count % 5 == 0:
                            log_info(f"Scroll {scroll_count} : {last_count} annonces collectees...")

                    log_info(f"{len(found_hrefs)} annonces collectees jusqu'ici.")
                except Exception as e:
                    log_warn(f"Erreur sur URL {url}: {e}")

            browser.close()
            
    except Exception as e:
        log_error(f"Erreur Playwright Critique : {e}")
        
    if not found_items_data:
        log_warn("Aucune annonce trouvee.")
        return 0

    # 2. Visiter chaque annonce en parallele.
    log_info(f"Analyse de {len(found_items_data)} annonces en parallele...")
    if found_items_data:
        log_info(f"Exemple de liens : {[item['href'] for item in found_items_data[:3]]}")
    
    final_images_to_download = []
    session = requests.Session()
    session.headers.update({
        "User-Agent": ua.random,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
    })
    
    def process_item(item):
        try:
            r = session.get(item["href"], timeout=15)
            if r.status_code != 200:
                return []
            
            soup = BeautifulSoup(r.text, "html.parser")
            page_title = ""
            if soup.title and soup.title.string:
                page_title = soup.title.string.strip()
            meta_desc = soup.find("meta", attrs={"name": "description"})
            meta_text = meta_desc.get("content", "") if meta_desc else ""
            item_title = page_title or item.get("title", "")
            grade = extract_grade(f"{item_title} {meta_text}") or item.get("grade", "unknown")

            item_image_urls = []
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src")
                src_hd = normalize_gcc_image_url(src)
                if not src_hd:
                    continue
                if not any(token in src_hd.lower() for token in ("item_recto", "item_verso", "/item/")):
                    continue
                if src_hd not in item_image_urls:
                    item_image_urls.append(src_hd)

            res = []
            for idx, img_url in enumerate(item_image_urls[:2]):
                res.append({
                    "url": img_url,
                    "grade": grade,
                    "side": image_side_from_url(img_url, idx),
                    "item_id": item["href"].rstrip("/").rsplit("/", 1)[-1],
                })
            return res
        except Exception:
            return []

    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(process_item, item): item for item in found_items_data}
        for future in as_completed(future_to_url):
            res = future.result()
            if res:
                final_images_to_download.extend(res)
                if len(final_images_to_download) >= max_images:
                    break

    log_info(f"Preparation du telechargement de {len(final_images_to_download)} images en HD RAW...")
    
    # 3. Telechargement en PARALLELE
    count = 0
    def download_task(item, idx):
        img_url = item["url"]
        grade = item["grade"]
        side = item["side"]
        item_id = safe_filename_part(item.get("item_id", "item"))
        
        grade_folder = os.path.join(save_folder, grade, side)
        os.makedirs(grade_folder, exist_ok=True)
        
        ext = os.path.splitext(urlparse(img_url).path)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"
        filepath = os.path.join(grade_folder, f"gcc_{item_id}_{side}_{idx}{ext}")
        
        if os.path.exists(filepath):
            return False

        success, img_hash = download_image(img_url, filepath, session)
        if success:
            if img_hash in existing_hashes:
                try: os.remove(filepath)
                except: pass
                return False
            existing_hashes.add(img_hash)
            log_success(f"Telecharge : {grade}/{side} - {os.path.basename(filepath)}")
            return True
        return False

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(download_task, item, i) for i, item in enumerate(final_images_to_download)]
        for future in as_completed(futures):
            if future.result():
                count += 1
                if count >= max_images:
                    break

    log_success(f"Graded Card Center : {count} images recuperees !")
    return count


# ============================================================================
#                 SOURCE 2b : EBAY SIMPLE (fallback sans Scrapy)
# ============================================================================

def scrape_ebay_simple(max_images, existing_hashes, save_folder=SAVE_FOLDER):
    """
    Scrape eBay avec requests + BeautifulSoup (fallback).
    Moins efficace que Scrapy mais ne necessite pas de dep supplementaire.
    """
    log_header("SOURCE : eBay (simple)")

    count = 0
    session = requests.Session()
    # Pre-warm session avec la homepage
    try:
        session.get("https://www.ebay.com", headers=get_headers(), timeout=10)
        time.sleep(2)
    except Exception:
        pass

    for query in SEARCH_QUERIES:
        if count >= max_images:
            break

        for page in range(1, 4):
            if count >= max_images:
                break

            log_info(f"eBay '{query}' - Page {page} ({count}/{max_images})")

            url = f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(query)}&_pgn={page}&_ipg=100"

            try:
                response = session.get(
                    url,
                    headers=get_headers(),
                    proxies=get_proxy(),
                    timeout=15
                )

                if response.status_code == 403:
                    log_warn(f"403 Forbidden - eBay bloque (page {page})")
                    time.sleep(10)
                    continue

                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                img_urls = set()

                # Selecteurs multiples
                for img_tag in soup.find_all("img"):
                    img_url = (
                        img_tag.get("data-src") or
                        img_tag.get("src") or
                        ""
                    )
                    if img_url and "http" in img_url and "s-l" in img_url:
                        img_url_hd = img_url.split('?')[0]
                        for low_res in ["s-l225", "s-l300", "s-l140", "s-l64", "s-l96"]:
                            img_url_hd = img_url_hd.replace(low_res, "s-l1600")
                        img_urls.add(img_url_hd)

                for img_url in img_urls:
                    if count >= max_images:
                        break

                    filepath = os.path.join(save_folder, f"ebay_simple_{count}.jpg")
                    success, img_hash = download_image(img_url, filepath, session)

                    if success:
                        if img_hash in existing_hashes:
                            os.remove(filepath)
                            continue
                        existing_hashes.add(img_hash)
                        count += 1

                time.sleep(random.uniform(3.0, 6.0))

            except Exception as e:
                log_error(f"Erreur page {page} : {e}")
                time.sleep(5)

    log_success(f"eBay simple : {count} images recuperees !")
    return count


# ============================================================================
#                     SOURCE 3 : GOOGLE IMAGES (PLAYWRIGHT)
# ============================================================================

def scrape_google_images(max_images, existing_hashes, save_folder=SAVE_FOLDER):
    """
    Scrape Google Images en utilisant Playwright pour recuperer des images de slabs.
    """
    log_header("SOURCE : Google Images (Playwright)")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log_error("Playwright n'est pas installe. Source Google desactivee.")
        return 0

    found_image_urls = []
    
    log_info("Demarrage du navigateur Playwright...")
    try:
        with sync_playwright() as p:
            # Lancer Chromium en mode non-headless pour eviter la detection
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={'width': 1920, 'height': 1080}
            )
            page = context.new_page()
            
            for query in SEARCH_QUERIES:
                if len(found_image_urls) >= max_images:
                    break
                    
                log_info(f"Playwright: Google Images '{query}'")
                url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch"
                
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    
                    # Clic sur le bouton "Accepter tout" si present (cookies Google)
                    try:
                        page.click('button:has-text("Tout accepter")', timeout=3000)
                    except:
                        pass
                        
                    # Scroll pour charger plus d'images
                    for _ in range(3):
                        page.evaluate("window.scrollBy(0, 10000)")
                        page.wait_for_timeout(1000)
                    
                    # Recuperer les images
                    img_elements = page.query_selector_all("img")
                    for img in img_elements:
                        if len(found_image_urls) >= max_images:
                            break
                            
                        src = img.get_attribute("src")
                        data_src = img.get_attribute("data-src")
                        img_url = data_src or src or ""
                        
                        if img_url and img_url.startswith("http") and "gstatic" not in img_url and "google" not in img_url:
                            if img_url not in found_image_urls:
                                found_image_urls.append(img_url)
                                
                except Exception as e:
                    log_warn(f"Erreur sur la recherche Google: {e}")
                    continue
                    
            browser.close()
            
    except Exception as e:
        log_error(f"Erreur Playwright : {e}")
        
    log_info(f"Playwright a trouve {len(found_image_urls)} URLs d'images Google")
    
    count = 0
    session = requests.Session()

    for i, img_url in enumerate(found_image_urls):
        if count >= max_images:
            break

        filepath = os.path.join(save_folder, f"google_{int(time.time())}_{i}.jpg")
        success, img_hash = download_image(img_url, filepath, session, min_size_kb=8)

        if success:
            if img_hash in existing_hashes:
                os.remove(filepath)
                continue
            existing_hashes.add(img_hash)
            count += 1
            if count % 10 == 0:
                log_success(f"{count} images Google telechargees...")

        time.sleep(random.uniform(0.3, 1.0))

    log_success(f"Google Playwright : {count} images recuperees !")
    return count

# ============================================================================
#                     SOURCE 4 : POKELLECTOR
# ============================================================================

def scrape_pokellector(max_images, existing_hashes, save_folder=SAVE_FOLDER):
    """
    Scrape pokellector.com pour des scans de cartes en haute qualite.
    """
    log_header("SOURCE : Pokellector")

    count = 0
    session = requests.Session()
    base_url = "https://www.pokellector.com"

    try:
        log_info("Recuperation de la liste des sets...")
        response = session.get(f"{base_url}/sets", headers=get_headers(), timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        set_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "-Expansion/" in href or "/sets/" in href:
                full_url = urljoin(base_url, href)
                if full_url not in set_links:
                    set_links.append(full_url)

        log_info(f"Trouve {len(set_links)} liens de sets/cartes")

        for set_url in set_links[:30]:
            if count >= max_images:
                break

            try:
                resp = session.get(set_url, headers=get_headers(), timeout=15)
                set_soup = BeautifulSoup(resp.text, "html.parser")

                for img in set_soup.find_all("img"):
                    if count >= max_images:
                        break

                    src = img.get("data-src") or img.get("src") or ""

                    if not src.startswith("http"):
                        continue
                    if any(x in src.lower() for x in ["logo", "banner", "icon", "avatar", "sprite"]):
                        continue

                    filepath = os.path.join(save_folder, f"pokellector_{count}.jpg")
                    success, img_hash = download_image(src, filepath, session)

                    if success:
                        if img_hash in existing_hashes:
                            os.remove(filepath)
                            continue
                        existing_hashes.add(img_hash)
                        count += 1

                time.sleep(random.uniform(1.0, 3.0))

            except Exception as e:
                log_error(f"Erreur set : {e}")
                continue

    except Exception as e:
        log_error(f"Erreur Pokellector : {e}")

    log_success(f"Pokellector : {count} images recuperees !")
    return count


# ============================================================================
#                           MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PokeScan Scrapper - Recupere des images de cartes Pokemon gradees",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sources disponibles:
  gcc          Graded Card Center (super fiable, images de slabs PSA/PCA)
  tcg          API PokemonTCG (images HD classiques sans slab)
  google       Google Images (cartes gradées)
  pokellector  Pokellector.com
  all          Alias de gcc dans ce projet local

Exemples:
  python scrapper.py --sources gcc --max 200
  python scrapper.py --sources gcc google --max 500
  python scrapper.py --sources all --max 1000
        """
    )
    parser.add_argument("--max", type=int, default=DEFAULT_MAX_IMAGES,
                        help=f"Nombre max d'images a telecharger (defaut: {DEFAULT_MAX_IMAGES})")
    parser.add_argument("--sources", nargs="+", default=["gcc"],
                        choices=["gcc", "tcg", "google", "pokellector", "all"],
                        help="Sources a utiliser (defaut: gcc)")
    parser.add_argument("--folder", type=str, default=SAVE_FOLDER,
                        help=f"Dossier de sauvegarde (defaut: {SAVE_FOLDER})")

    args = parser.parse_args()

    save_folder = args.folder
    max_images = args.max
    sources = args.sources

    if "all" in sources:
        sources = ["gcc"]

    # Creer le dossier
    os.makedirs(save_folder, exist_ok=True)

    # Banner
    print()
    log("""
    +======================================================+
    |                                                      |
    |   POKESCAN SCRAPPER v2.0                             |
    |   Recuperation d'images de cartes Pokemon            |
    |                                                      |
    +======================================================+
    """, Colors.HEADER)

    log_info(f"Dossier de sauvegarde : {save_folder}")
    log_info(f"Objectif : {max_images} images")
    log_info(f"Sources : {', '.join(sources)}")

    # Compter les images existantes, y compris dataset_pokemon/<grade>/<side>/.
    existing_count = count_local_images(save_folder)
    if existing_count > 0:
        log_info(f"Images existantes dans le dossier : {existing_count}")

    # Hashes pour deduplication
    existing_hashes = set()
    if existing_count:
        log_info("Calcul des hash des images existantes...")
        existing_hashes = collect_existing_hashes(save_folder)
        log_info(f"{len(existing_hashes)} hash calcules")

    # Repartir le quota entre les sources
    per_source = max_images // len(sources)
    remainder = max_images % len(sources)

    total_downloaded = 0
    start_time = time.time()

    # Mapping des sources
    source_functions = {
        "gcc": scrape_gcc_playwright,
        "tcg": scrape_pokemon_tcg_api,
        "google": scrape_google_images,
        "pokellector": scrape_pokellector,
    }

    for i, source in enumerate(sources):
        quota = per_source + (1 if i < remainder else 0)
        func = source_functions.get(source)
        if func:
            downloaded = func(quota, existing_hashes, save_folder)
            total_downloaded += downloaded

    # Resume final
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    log_header("RESUME")
    log_success(f"Total images telechargees : {total_downloaded}")
    log_info(f"Temps ecoule : {minutes}m {seconds}s")
    log_info(f"Dossier : {os.path.abspath(save_folder)}")

    final_count = count_local_images(save_folder)
    log_info(f"Total images dans le dossier : {final_count}")

    if total_downloaded == 0:
        print()
        log_warn("Aucune image telechargee !")
        log_warn("Conseils :")
        log_warn("  - Verifie ta connexion internet")
        log_warn("  - Relance GCC seul : python scrapper.py --sources gcc --max 20")
        log_warn("  - Lance playwright install chromium si le navigateur manque")
        log_warn("  - Baisse le --max pour tester rapidement")
    else:
        print()
        log_success("Scraping termine avec succes !")
        log_info("Prochaine etape : lance l'entrainement avec pokescan_amd.py")


if __name__ == "__main__":
    main()
