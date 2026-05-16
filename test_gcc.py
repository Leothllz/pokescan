from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('https://gradedcardcenter.com/filtres/fixed-price?itemTypes=%5B%22CARDS%22%5D&categories=%5B%22Pokemon%22%5D&gradingCompanies=%5B%22PSA%22%2C%22PCA%22%5D')
        page.wait_for_timeout(3000)
        items = page.query_selector_all('a')
        for item in items:
            href = item.get_attribute('href')
            if href and '/item/' in href:
                # Get parent container
                parent = item.evaluate_handle('el => el.parentElement')
                print(parent.inner_html())
                break
        browser.close()
run()
