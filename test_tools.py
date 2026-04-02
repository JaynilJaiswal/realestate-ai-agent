# test_tools.py
from backend.agent.tools import search_live_market_data, extract_text_from_url, calculate_math_expression

def test_pipeline():
    print("=== TESTING OPTIMIZED TOOLS ===\n")

    # Test 1: Targeted Scraping
    print("1. Testing Semantic URL Extraction...")
    test_url = "https://en.wikipedia.org/wiki/Real_estate"
    scrape_result = extract_text_from_url.invoke({"url": test_url})
    
    print(f"Scraped {len(scrape_result)} characters of dense content from {test_url}.\n")
    print("Preview of scraped text (Should contain no navbars/menus):")
    print(scrape_result[:4000] + "...\n")
    print("="*50 + "\n")

    # Test 2: Generic Math Evaluator
    print("2. Testing Generic Math Evaluator...")
    
    # Example 1: Simple percentage (20% down payment on $350k)
    simple_math = "350000 * 0.20"
    res_1 = calculate_math_expression.invoke({"expression": simple_math})
    print(f"Down Payment ({simple_math}) -> {res_1}")
    
    # Example 2: Complex Mortgage Formula
    # Principal: 280000, Rate: 6.5%, Years: 30
    mortgage_math = "280000 * (0.065/12) / (1 - (1 + 0.065/12)**-360)"
    res_2 = calculate_math_expression.invoke({"expression": mortgage_math})
    print(f"Mortgage Payment ({mortgage_math}) -> {res_2}\n")

if __name__ == "__main__":
    test_pipeline()