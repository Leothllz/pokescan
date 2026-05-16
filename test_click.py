from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('https://gradedcardcenter.com/filtres/fixed-price?itemTypes=%5B%22CARDS%22%5D&categories=%5B%22Pokemon%22%5D&gradingCompanies=%5B%22PSA%22%2C%22PCA%22%5D')
        page.wait_for_timeout(3000)
        
        # Click the first item
        page.click('a[href*="/item/"]')
        page.wait_for_timeout(3000)
        
        print("URL:", page.url)
        imgs = page.query_selector_all('img')
        for img in imgs:
            src = img.get_attribute('src')
            if src and 'cdn.gradedcardcenter.com' in src:
                print("IMG:", src)
                
        browser.close()
run()
