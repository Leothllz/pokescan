from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('https://gradedcardcenter.com/filtres/fixed-price?itemTypes=%5B%22CARDS%22%5D&categories=%5B%22Pokemon%22%5D&gradingCompanies=%5B%22PSA%22%2C%22PCA%22%5D')
        page.wait_for_timeout(3000)
        
        imgs = page.query_selector_all('img[src*="cdn.gradedcardcenter.com/cdn-cgi/image"]')
        for img in imgs[:5]:
            # find closest parent with some text
            parent = img.evaluate_handle('el => { let p = el.parentElement; while(p && p.innerText.trim() === "") { p = p.parentElement; } return p; }')
            if parent:
                print("--- TEXT ---")
                print(repr(parent.evaluate('el => el.innerText')))
                print("IMG:", img.get_attribute('src'))
        browser.close()
run()
