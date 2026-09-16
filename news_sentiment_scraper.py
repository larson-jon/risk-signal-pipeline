# NEWS SENTIMENT SCRAPER - CLEAN VERSION
# single script for enterprise and emerging risks
# decodes search terms, fetches via gnews.io API, analyzes sentiment, appends to CSV
# paywalled articles are included (even if parsing fails - sentiment neutral)

import datetime as dt
import random
import time
import re
import csv
import requests
from pathlib import Path
from newspaper import Article, Config
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.parse import urlparse
import pandas as pd
from dateutil import parser
import sys
import argparse
import os
import json

# SSL Certificate setup
def setup_ssl_verification():
    """Setup SSL verification based on environment"""
    cert_path = Path(__file__).parent / "combined-certs.pem"

    # Check if running in GitHub Actions
    if os.getenv('GITHUB_ACTIONS'):
        print("Running in GitHub Actions - using default SSL verification")
        return True  # Use default verification

    # Check if corporate cert exists (local FINRA environment)
    if cert_path.exists():
        print(f"Using corporate certificate: {cert_path}")
        return str(cert_path)

    # Fallback - disable verification with warning
    print("⚠️  No certificate found - SSL verification disabled")
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return False

# GLOBAL CONSTANTS
SEARCH_DAYS = 7  # look back this many days for news articles
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
MAX_ARTICLES_PER_TERM = 10
MAX_SEARCH_TERMS = 5 if DEBUG_MODE else None  # limit terms in debug for quick testing
VERIFY_SSL = setup_ssl_verification()

# decoding logic - kept from original
def process_encoded_search_terms(term):
    try:
        encoded_number = int(term)
        byte_length = (encoded_number.bit_length() + 7) // 8
        byte_rep = encoded_number.to_bytes(byte_length, byteorder='little')
        decoded_text = byte_rep.decode('utf-8')
        return decoded_text
    except (ValueError, UnicodeDecodeError, OverflowError):
        return None

# simple session with retries and user agents - merged from utils
class ScraperSession:
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        ]
    
    def get_random_headers(self):
        return {'User-Agent': random.choice(self.user_agents)}

# extract domain name - kept similar to original utils
def get_source_name(url):
    domain = urlparse(url).netloc.replace('www.', '')
    parts = domain.split('.')
    if len(parts) > 2:
        return parts[-2]  # e.g., bloomberg from bloomberg.com
    return domain

# setup output dir and path
def setup_output_dir(output_csv):
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    return output_dir / output_csv

# load existing links for dedup - simple version
def load_existing_links(csv_path):
    if DEBUG_MODE:
        print("DEBUG: Skipping existing links load")
        return set()
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=['LINK'])
        links = set(df['LINK'].dropna().str.lower().str.strip())
        print(f"Loaded {len(links)} existing links")
        return links
    except Exception as e:
        print(f"Warning: Could not load existing links: {e}")
        return set()

# save results with append and dedup - fixed for empty csv on ubuntu/actions
def save_results(df, output_path):
    print(f"DEBUG save: df shape {df.shape}, empty={df.empty}, path={output_path}")
    
    output_dir = output_path.parent
    output_dir.mkdir(exist_ok=True)
    print(f"DEBUG dir exists: {output_dir.exists()}, writable: {os.access(str(output_dir), os.W_OK)}")
    
    try:
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['LINK'], keep='first')  # Use LINK only for dedup
        else:
            combined_df = df
        
        # Only write if there's data (handles initial empty case)
        if not combined_df.empty:
            combined_df.sort_values(by='PUBLISHED_DATE', ascending=False).to_csv(
                output_path, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL
            )
        else:
            print("No data to save - skipping write to avoid empty file.")
        
        # Verify
        if output_path.exists():
            size = output_path.stat().st_size
            print(f"SUCCESS: CSV updated/created, size {size}B")
            total_records = len(pd.read_csv(output_path))
            return total_records
        else:
            print("ERROR: File missing after save!")
            return 0
            
    except Exception as e:
        print(f"ERROR in save_results: {e}")
        import traceback
        traceback.print_exc()
        return 0

# argparse setup - kept original
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--risk-type', type=str, choices=['enterprise', 'emerging'], required=True)
args = arg_parser.parse_args()

def main():
    risk_type = args.risk_type
    if risk_type == "enterprise":
        risk_id_col = "ENTERPRISE_RISK_ID"
        encoded_csv = "data/EnterpriseRisksListEncoded.csv"
        output_csv = "enterprise_risks_online_sentiment.csv"
    else:
        risk_id_col = "EMERGING_RISK_ID"
        encoded_csv = "data/EmergingRisksListEncoded.csv"
        output_csv = "emerging_risks_online_sentiment.csv"
    
    print("#" * 50)
    start_time = dt.datetime.now()
    print(f"Starting {risk_type.upper()} News Sentiment Scraper")
    print(f"DEBUG_MODE: {DEBUG_MODE}")
    print(f"Script started: {start_time}")
    print("#" * 50)
    
    # setup
    session = ScraperSession()
    analyzer = SentimentIntensityAnalyzer()

    # Configure SSL for transformers/huggingface downloads
    if VERIFY_SSL and VERIFY_SSL != True:
        os.environ['REQUESTS_CA_BUNDLE'] = VERIFY_SSL
        os.environ['SSL_CERT_FILE'] = VERIFY_SSL
        os.environ['CURL_CA_BUNDLE'] = VERIFY_SSL

    
    config = Config()
    config.fetch_images = False
    config.memoize_articles = False
    config.request_timeout = 30

    # Set SSL certificate for newspaper library
    if VERIFY_SSL and VERIFY_SSL != True:  # If using custom cert path
        os.environ['REQUESTS_CA_BUNDLE'] = VERIFY_SSL
        os.environ['SSL_CERT_FILE'] = VERIFY_SSL
    
    output_path = setup_output_dir(output_csv)
    existing_links = load_existing_links(output_path)
    
    # load and decode search terms - close to original
    try:
        usecols = [risk_id_col, 'SEARCH_TERM_ID', 'ENCODED_TERMS']
        df = pd.read_csv(encoded_csv, usecols=usecols)
        df[risk_id_col] = pd.to_numeric(df[risk_id_col], downcast='integer', errors='coerce')
        df['SEARCH_TERMS'] = df['ENCODED_TERMS'].apply(process_encoded_search_terms)
        
        valid_df = df.dropna(subset=['SEARCH_TERMS'])
        print("\n" + "=" * 50)
        print("DECODED SEARCH TERMS DEBUG")
        print("=" * 50)
        for idx, row in valid_df.iterrows():
            print(f"Risk ID: {row[risk_id_col]} | Term ID: {row['SEARCH_TERM_ID']}")
            print(f"  Encoded: {row['ENCODED_TERMS']}")
            print(f"  Decoded: '{row['SEARCH_TERMS']}'")
            print(f"  Length: {len(row['SEARCH_TERMS'])} chars")
            print()
        if valid_df.empty:
            print("ERROR: No valid search terms after decoding!")
            sys.exit(1)
        
        print(f"Loaded {len(valid_df)} valid search terms")
        if DEBUG_MODE:
            print("Sample decoded terms:", valid_df['SEARCH_TERMS'].head().tolist())
        
        # limit in debug
        if DEBUG_MODE and MAX_SEARCH_TERMS:
            valid_df = valid_df.head(MAX_SEARCH_TERMS)
            print(f"DEBUG: Limited to {MAX_SEARCH_TERMS} terms")
        
    except Exception as e:
        print(f"ERROR loading {encoded_csv}: {e}")
        sys.exit(1)

    # gnews.io API setup
    api_key = os.getenv('NEWS_DATA_API_KEY')
    if not api_key:
        print("WARNING: NEWS_DATA_API_KEY - skipping fetch")
        articles_df = pd.DataFrame(columns=[
            'RISK_ID', 'SEARCH_TERM_ID', 'GOOGLE_INDEX', 'TITLE', 'LINK', 'PUBLISHED_DATE',
            'SUMMARY', 'KEYWORDS', 'SENTIMENT_COMPOUND', 'SENTIMENT', 'SOURCE', 'QUALITY_SCORE'
        ])
    else:
        base_url = "https://newsdata.io/api/1/news"
        all_articles = []

        # fetch articles for each search term
        for idx, row in valid_df.iterrows():
            risk_id = row[risk_id_col]
            search_term_id = row['SEARCH_TERM_ID']
            search_term = row['SEARCH_TERMS'].strip()  # remove trailing spaces

            print(f"processing term {idx+1}/{len(valid_df)}: risk_id={risk_id}, term_id={search_term_id} - '{search_term}'")

            # calculate from date for lookback
            today = dt.date.today()
            from_date = today - dt.timedelta(days=SEARCH_DAYS)
            from_date_str = from_date.isoformat()
    
            params = {
                'q': search_term,
                'language': 'en',
                'country': 'us',
                'size': MAX_ARTICLES_PER_TERM,
                'apikey': api_key
                
            }
    
            if DEBUG_MODE:
                print(f"  Using date range: from {from_date_str}")

            try:
                articles = []
                next_page = None
                while len(articles) < MAX_ARTICLES_PER_TERM:
                    if next_page:
                        params['page'] = next_page
                    else:
                        params.pop('page', None)
                    
                    response = requests.get(base_url, params=params, timeout=30, verify=VERIFY_SSL)
                    if response.status_code != 200:
                        print(f"  api error {response.status_code}: {response.text}")
                        break
                    data = response.json()
                    new_articles = data.get('results', [])
                    articles.extend(new_articles)
                    if 'nextPage' in data:
                        next_page = data['nextPage']
                    else:
                        break
                    if len(new_articles) < params.get('size', 10):
                        break
                    time.sleep(1)  # Rate limit
                    
                articles = articles[:MAX_ARTICLES_PER_TERM]
                print(f"  found {len(articles)} articles")

                for google_index, item in enumerate(articles, start=1):
                    url = item['link']
                    title = item['title']
                    published_date = item.get('pubDate', '')
                    source_name = item.get('source_id', get_source_name(url))

                    # dedup by url
                    if url.lower().strip() in existing_links:
                        if DEBUG_MODE:
                            print(f"  skipping duplicate url: {title[:50]}...")
                        continue

                    # parse article
                    summary = ''
                    keywords = ''
                    sentiment_compound = 0.0
                    sentiment = 'Neutral'

                    try:
                        article = Article(url, config=config)
                        article.download()
                        if article.download_state == 2:  # success
                            article.parse()
                            summary = article.summary if article.summary else (article.text[:500] if article.text else '')

                            if article.keywords:
                                keywords = ', '.join(article.keywords)
                            else:
                                # Simple fallback: extract most common words from title
                                from collections import Counter
                                import re
                                words = re.findall(r'\b[a-z]{4,}\b', (article.title or '').lower())
                                common_words = Counter(words).most_common(5)
                                keywords = ', '.join([word for word, _ in common_words]) if common_words else ''
                            
                            # sentiment
                            sentiment_text = title + " " + summary
                            sentiment_scores = analyzer.polarity_scores(sentiment_text)
                            sentiment_compound = sentiment_scores['compound']
                            if sentiment_compound >= 0.05:
                                sentiment = 'Positive'
                            elif sentiment_compound <= -0.05:
                                sentiment = 'Negative'
                        else:
                            if DEBUG_MODE:
                                print(f"  download failed for '{title[:50]}...' (likely paywalled)")
                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"  parsing error for '{title[:50]}...': {e}")

                    # record - always include even if parsing failed
                    article_record = {
                        'RISK_ID': risk_id,
                        'SEARCH_TERM_ID': search_term_id,
                        'GOOGLE_INDEX': google_index,
                        'TITLE': title,
                        'LINK': url,
                        'PUBLISHED_DATE': published_date,
                        'SUMMARY': summary[:500],  # truncate for csv size
                        'KEYWORDS': keywords,
                        'SENTIMENT_COMPOUND': round(sentiment_compound, 4),
                        'SENTIMENT': sentiment,
                        'SOURCE': source_name,
                        'QUALITY_SCORE': 0
                    }

                    all_articles.append(article_record)
                    existing_links.add(url.lower().strip())

                    if DEBUG_MODE:
                        print(f"  added: {title[:60]}... ({source_name}) | sentiment: {sentiment}")

            except Exception as e:
                print(f"  error fetching for term '{search_term}': {e}")

            # polite delay
            time.sleep(1 if DEBUG_MODE else 3)

        # build final dataframe
        columns = ['RISK_ID', 'SEARCH_TERM_ID', 'GOOGLE_INDEX', 'TITLE', 'LINK', 'PUBLISHED_DATE',
                   'SUMMARY', 'KEYWORDS', 'SENTIMENT_COMPOUND', 'SENTIMENT', 'SOURCE', 'QUALITY_SCORE']
        articles_df = pd.DataFrame(all_articles, columns=columns)

    print(f"Collected and processed {len(articles_df)} new articles")

    record_count = save_results(articles_df, output_path)
    
    # debug list output after save
    print("DEBUG final output ls:")
    os.system("ls -la output/ || echo 'output empty'")
    
    if record_count == 0 and len(articles_df) == 0:
        print(f"Created new empty output file with headers: {output_path}")
    else:
        print(f"Updated output file - total records: {record_count}")

    print(f"Completed at: {dt.datetime.now()}")
    print("#" * 50)

if __name__ == '__main__':
    main()
