from stock_analyser.scrapers.test_scraper import scrape_test


def test_scrape(request, symbol):
    return scrape_test(symbol)