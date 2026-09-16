"""
Test API call with a specific search term(s)
"""

import requests
import sys
import os
from pathlib import Path


def test_api_search(search_term):
    api_key = os.getenv('NEWS_DATA_API_KEY')
    if not api_key:
        print("❌ ERROR: NEWS_DATA_API_KEY not set")
        return

    print(f"\n{'=' * 70}")
    print(f"Testing API Search")
    print(f"{'=' * 70}")
    print(f"Search term: '{search_term}'")

    base_url = "https://newsdata.io/api/1/news"

    params = {
        'q': search_term,
        'language': 'en',
        'country': 'us',
        'size': 10,
        'apikey': api_key
    }

    # Path to combined certificate bundle
    cert_path = Path(__file__).parent / "combined-certs.pem"

    if not cert_path.exists():
        print(f"⚠️  Certificate file not found: {cert_path}")
        print(f"   Falling back to verify=False")
        verify_param = False
    else:
        print(f"✅ Using certificate: {cert_path}")
        verify_param = str(cert_path)

    print(f"\nSending request...")

    try:
        response = requests.get(base_url, params=params, timeout=30, verify=verify_param)

        print(f"Status code: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ API Error: {response.text}")
            return

        data = response.json()

        print(f"\n📊 API Response:")
        print(f"   Status: {data.get('status', 'unknown')}")
        print(f"   Total results: {data.get('totalResults', 0)}")

        results = data.get('results', [])
        print(f"   Results returned: {len(results)}")

        if len(results) == 0:
            print(f"\n⚠️  ZERO RESULTS for '{search_term}'")
            print(f"   This term may be too specific or have no recent news")
        else:
            print(f"\n✅ Found {len(results)} articles!")
            print(f"\nFirst 3 results:")
            for i, article in enumerate(results[:3], 1):
                print(f"\n   {i}. {article.get('title', 'No title')}")
                print(f"      Source: {article.get('source_id', 'unknown')}")
                print(f"      Date: {article.get('pubDate', 'unknown')}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    search_term = sys.argv[1] if len(sys.argv) > 1 else "artificial intelligence"
    test_api_search(search_term)
