#!/usr/bin/env python3
"""
Standalone News Scraper for German/Austrian Trade Publications
Works independently without Scrapy or existing project dependencies
"""

import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Tuple
import logging
import uuid
from bs4 import BeautifulSoup
import re
from openai import OpenAI
import openai
import requests
import time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
HISTORICAL_MODE = False  # Set to True for initial historical scraping
TEST_MODE = False  # Set to True for testing without database writes
PROXY_TIMEOUT = 30

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_sentiment(text: str) -> dict[str, float]:
    """

    :param text:
    :return:
    {
        pos: float,
        neg: float,
        neu: float
    }
    """
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(text)
        return sentiment_scores  # type: ignore
    except Exception as e:
        return {'error': str(e)}

def generate_completion_keywords_topics(prompt_input: str) -> Optional[str]:
    cl = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    count = 2
    while True:
        count += 1
        if count > 5:
            print(f"Too many retries and no success Num retries {count}")

            print(f"Retrying in 300 seconds")
            time.sleep(300)

        try:
            prompt = """
From news title/description, extract:
- Max 5 keywords (multi-word OK, by relevance, no duplicates)
- Max 5 topics (lowercase words like technology, politics, business; family-friendly)

Format: {"keywords":"keyword1,keyword2","topics":"topic1,topic2"}

Example: "Tesla Unveils New Electric Truck" → {"keywords":"tesla electric truck,500-mile range,battery technology","topics":"technology,automotive,business"}
"""
            completion = cl.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user","content": prompt_input}
                ]
            )

            output = completion.choices[0].message.content
            return output
        except openai.RateLimitError as e:
            print(f"Rate limit exceeded. Retrying in {2**count} seconds. error: {e}")
            time.sleep(count)
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return None

class ConfigManager:
    """Manages configuration loading and source filtering"""
    
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file {self.config_path} not found")
            return {"sources": []}
    
    def get_sources_by_shard(self, shard_color: str) -> List[Dict]:
        """Filter sources by shard color"""
        return [
            source for source in self.config.get('sources', [])
            if source.get('shard_color') == shard_color and source.get('enabled', True)
        ]
    
    def get_source_by_name(self, source_name: str) -> Optional[Dict]:
        """Get a specific source by name"""
        for source in self.config.get('sources', []):
            if source.get('name') == source_name:
                return source
        return None

class SitemapHandler:
    """Base class for sitemap handlers"""
    
    def __init__(self, session: requests.Session = None):
        self.session = session or requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_article_urls(self, source_config: Dict) -> List[Tuple[str, str]]:
        """Get article URLs and their last modified dates"""
        raise NotImplementedError

class StandardSitemapHandler(SitemapHandler):
    """Handler for standard XML sitemaps"""
    
    def get_article_urls(self, source_config: Dict) -> List[Tuple[str, str]]:
        """Extract URLs from standard XML sitemaps"""
        urls = []
        
        for sitemap_url in source_config.get('sitemap_urls', []):
            try:
                logger.info(f"Fetching sitemap: {sitemap_url}")
                response = self.session.get(sitemap_url, timeout=PROXY_TIMEOUT)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                
                # Handle different XML namespaces
                namespaces = {'': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                
                for url_elem in root.findall('.//url', namespaces):
                    loc_elem = url_elem.find('loc', namespaces)
                    lastmod_elem = url_elem.find('lastmod', namespaces)
                    
                    if loc_elem is not None:
                        url = loc_elem.text
                        lastmod = lastmod_elem.text if lastmod_elem is not None else datetime.now().isoformat()
                        urls.append((url, lastmod))
                
                logger.info(f"Found {len(urls)} URLs in sitemap: {sitemap_url}")
                
            except Exception as e:
                logger.error(f"Error processing sitemap {sitemap_url}: {e}")
        
        return urls

class DateDynamicSitemapHandler(SitemapHandler):
    """Handler for date-dynamic sitemaps"""
    
    def get_article_urls(self, source_config: Dict) -> List[Tuple[str, str]]:
        """Extract URLs from date-dynamic sitemaps"""
        urls = []
        pattern = source_config.get('sitemap_pattern', '')
        fallback_months = source_config.get('fallback_months', 2)
        
        # Generate date parameters
        current_date = datetime.now()
        dates_to_try = []
        
        if HISTORICAL_MODE:
            # For historical mode, go back 6 months
            for i in range(6):
                date = current_date - timedelta(days=30 * i)
                dates_to_try.append(date.strftime('%Y_%m'))
        else:
            # For incremental mode, try current month and fallback months
            for i in range(fallback_months + 1):
                date = current_date - timedelta(days=30 * i)
                dates_to_try.append(date.strftime('%Y_%m'))
        
        for date_param in dates_to_try:
            sitemap_url = pattern.format(YYYY_MM=date_param)
            
            try:
                logger.info(f"Fetching dynamic sitemap: {sitemap_url}")
                response = self.session.get(sitemap_url, timeout=PROXY_TIMEOUT)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                namespaces = {'': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                
                month_urls = []
                for url_elem in root.findall('.//url', namespaces):
                    loc_elem = url_elem.find('loc', namespaces)
                    lastmod_elem = url_elem.find('lastmod', namespaces)
                    
                    if loc_elem is not None:
                        url = loc_elem.text
                        lastmod = lastmod_elem.text if lastmod_elem is not None else datetime.now().isoformat()
                        month_urls.append((url, lastmod))
                
                urls.extend(month_urls)
                logger.info(f"Found {len(month_urls)} URLs in {sitemap_url}")
                
            except Exception as e:
                logger.error(f"Error processing date-dynamic sitemap {sitemap_url}: {e}")
        
        return urls

class JetpackTableHandler(SitemapHandler):
    """Handler for Jetpack-style XML sitemaps with date filtering"""
    
    def get_article_urls(self, source_config: Dict) -> List[Tuple[str, str]]:
        """Extract URLs from Jetpack XML sitemaps, filtering by date"""
        urls = []
        
        for sitemap_url in source_config.get('sitemap_urls', []):
            try:
                logger.info(f"Fetching Jetpack XML sitemap: {sitemap_url}")
                response = self.session.get(sitemap_url, timeout=PROXY_TIMEOUT)
                response.raise_for_status()
                
                # Parse as XML (not HTML table as before)
                root = ET.fromstring(response.content)
                
                # Handle XML namespaces
                namespaces = {'': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                
                all_urls = []
                for url_elem in root.findall('.//url', namespaces):
                    loc_elem = url_elem.find('loc', namespaces)
                    lastmod_elem = url_elem.find('lastmod', namespaces)
                    
                    if loc_elem is not None:
                        url = loc_elem.text
                        lastmod = lastmod_elem.text if lastmod_elem is not None else datetime.now().isoformat()
                        
                        # Extract date from URL path (format: /2025/06/03/)
                        date_from_url = self._extract_date_from_url(url)
                        if date_from_url:
                            all_urls.append((url, lastmod, date_from_url))
                
                logger.info(f"Found {len(all_urls)} total URLs in Jetpack sitemap")
                
                # Filter by date - only get recent ones
                filtered_urls = self._filter_by_date(all_urls, source_config)
                
                # Convert back to the expected format (url, lastmod)
                urls.extend([(url, lastmod) for url, lastmod, _ in filtered_urls])
                
                logger.info(f"After date filtering: {len(filtered_urls)} URLs for processing")
                
            except Exception as e:
                logger.error(f"Error processing Jetpack sitemap {sitemap_url}: {e}")
        
        return urls
    
    def _extract_date_from_url(self, url: str) -> Optional[datetime]:
        """Extract date from URL path like /2025/06/03/"""
        try:
            # Look for pattern like /YYYY/MM/DD/ in URL
            import re
            pattern = r'/(\d{4})/(\d{2})/(\d{2})/'
            match = re.search(pattern, url)
            
            if match:
                year, month, day = match.groups()
                return datetime(int(year), int(month), int(day))
            
            return None
        except Exception as e:
            logger.warning(f"Could not extract date from URL {url}: {e}")
            return None
    
    def _filter_by_date(self, all_urls: List[Tuple[str, str, datetime]], source_config: Dict) -> List[Tuple[str, str, datetime]]:
        """Filter URLs by date, keeping only recent ones"""
        if not all_urls:
            return []
        
        # Get the latest publication date from Elasticsearch
        source_name = source_config.get('name', '')
        latest_es_date = self._get_latest_date_from_es(source_name)
        
        if HISTORICAL_MODE:
            # In historical mode, get last 6 months
            cutoff_date = datetime.now() - timedelta(days=180)
            logger.info(f"Historical mode: filtering URLs since {cutoff_date.date()}")
        else:
            # In incremental mode, only get URLs newer than latest in ES
            if latest_es_date:
                cutoff_date = latest_es_date
                logger.info(f"Incremental mode: filtering URLs newer than {cutoff_date.date()}")
            else:
                # If no date from ES, get last 7 days
                cutoff_date = datetime.now() - timedelta(days=7)
                logger.info(f"No ES date found: filtering URLs from last 7 days since {cutoff_date.date()}")
        
        # Filter URLs by date
        filtered_urls = []
        for url, lastmod, url_date in all_urls:
            if url_date and url_date >= cutoff_date:
                filtered_urls.append((url, lastmod, url_date))
        
        # Sort by date (newest first)
        filtered_urls.sort(key=lambda x: x[2], reverse=True)
        
        return filtered_urls
    
    def _get_latest_date_from_es(self, source_name: str) -> Optional[datetime]:
        """Get latest publication date from Elasticsearch for this source"""
        try:
            # This is a placeholder - you'll need to implement the actual ES query
            # For now, we'll use a mock function
            if TEST_MODE:
                # In test mode, return a date from 1 week ago for testing
                mock_date = datetime.now() - timedelta(days=7)
                logger.info(f"TEST MODE: Using mock ES date {mock_date.date()} for {source_name}")
                return mock_date
            else:
                # In production, you would call your actual ES function here
                # latest_date_str = get_latest_article_date(source_name)
                # return datetime.fromisoformat(latest_date_str.replace('Z', '+00:00'))
                
                # For now, return None to use fallback logic
                logger.warning(f"ES integration not implemented - using fallback date logic for {source_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest date from ES for {source_name}: {e}")
            return None

class CategoryPageHandler(SitemapHandler):
    """Handler for category pages without sitemaps"""
    
    def get_article_urls(self, source_config: Dict) -> List[Tuple[str, str]]:
        """Extract URLs from category listing pages"""
        urls = []
        max_pages = source_config.get('max_pages', 3)
        
        for category_url in source_config.get('category_urls', []):
            page_urls = self._extract_from_category_page(category_url, max_pages)
            urls.extend(page_urls)
        
        return urls
    
    def _extract_from_category_page(self, base_url: str, max_pages: int) -> List[Tuple[str, str]]:
        """Extract article URLs from a category page with pagination"""
        urls = []
        
        for page_num in range(1, max_pages + 1):
            if '?' in base_url:
                page_url = f"{base_url}&currPage={page_num}"
            else:
                page_url = f"{base_url}?currPage={page_num}"
            
            try:
                logger.info(f"Fetching category page {page_num}: {page_url}")
                response = self.session.get(page_url, timeout=PROXY_TIMEOUT)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                # print(soup)
                
                # Try different selectors for article links
                selectors = [
                    'a.teaser-link',
                ]
                
                article_links = []
                for selector in selectors:
                    links = soup.select(selector)
                    if links:
                        article_links = links
                        break
                
                page_urls = []
                for link in article_links:
                    href = link.get('href')
                    if href:
                        # Convert relative URLs to absolute
                        full_url = urljoin(base_url, href)
                        # Use current time as lastmod since category pages don't have this info
                        lastmod = datetime.now().isoformat()
                        page_urls.append((full_url, lastmod))
                
                urls.extend(page_urls)
                logger.info(f"Found {len(page_urls)} URLs on page {page_num}")
                
                # If no articles found, stop pagination
                if not page_urls:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing category page {page_url}: {e}")
                break
        
        return urls

class SitemapHandlerFactory:
    """Factory for creating appropriate sitemap handlers"""
    
    @staticmethod
    def create_handler(sitemap_type: str, session: requests.Session = None) -> SitemapHandler:
        """Create handler based on sitemap type"""
        handlers = {
            'standard': StandardSitemapHandler,
            'date_dynamic': DateDynamicSitemapHandler,
            'jetpack_table': JetpackTableHandler,
            'category_pages': CategoryPageHandler
        }
        
        handler_class = handlers.get(sitemap_type)
        if not handler_class:
            raise ValueError(f"Unknown sitemap type: {sitemap_type}")
        
        return handler_class(session)

class ContentExtractor:
    """Handles content extraction using BeautifulSoup (no newspaper3k dependency)"""
    
    def __init__(self, session: requests.Session = None):
        self.session = session or requests.Session()
        self.failed_domains = set()


    def _clean_promotional_content(self, content: str, title: str = None) -> str:
        """Remove specific promotional chunks from content"""
        if not content:
            return ""
        
        # Remove "Demnächst:" - make it more flexible
        content = re.sub(r'^\s*Demnächst:\s*', '', content, flags=re.IGNORECASE)
        # Also remove if it appears after some whitespace/newlines at start
        content = re.sub(r'^[\s\n]*Demnächst:\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^Mehr zu diesem Thema:\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^[\s\n]*Mehr zu diesem Thema:\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^\s*Demnächst:\s*', '', content, flags=re.IGNORECASE)
        # Also remove if it appears after some whitespace/newlines at start
        content = re.sub(r'^[\s\n]*Demnächst:\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^- Home - Newsticker -\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^[\s\n]*- Home - Newsticker -\s*', '', content, flags=re.IGNORECASE)
        # Remove title from beginning of content if it appears there
        if title and content.startswith(title):
            content = content[len(title):].strip()
            # Also remove common separators after title
            content = re.sub(r'^[:\-–—]\s*', '', content)
        
        # Remove title variations (sometimes with slight differences)
        if title:
            # Remove title if it appears at start with minor variations
            title_pattern = re.escape(title.strip())
            content = re.sub(rf'^{title_pattern}[:\-–—]?\s*', '', content, flags=re.IGNORECASE)
        
        
        # List of promotional chunks to remove completely
        promotional_chunks = [
            # Full chunk with title
            "Lebenswerte Lebensräume Die BKW bietet mit ihrem Netzwerk und ihrer Expertise zukunftsgerichtete Lösungen für Private und Unternehmen in den Bereichen Gebäude, Energie und Infrastruktur, um Wohlstand und Umwelt in einem lebenswerten Lebensraum in Einklang zu halten. Für Umgebungen, in denen Menschen gerne leben, gerne arbeiten und sich entfalten können.",
            # Just the main text without title (in case title appears separately)
            "Die BKW bietet mit ihrem Netzwerk und ihrer Expertise zukunftsgerichtete Lösungen für Private und Unternehmen in den Bereichen Gebäude, Energie und Infrastruktur, um Wohlstand und Umwelt in einem lebenswerten Lebensraum in Einklang zu halten. Für Umgebungen, in denen Menschen gerne leben, gerne arbeiten und sich entfalten können.",
            # Add more chunks here as you find them:
        ]
        
        # Standalone promotional titles/phrases to remove
        promotional_titles = [
            "Lebenswerte Lebensräume",
            # Add more standalone titles here:
        ]
        
        cleaned_content = content
        
        # Remove promotional chunks first
        for chunk in promotional_chunks:
            cleaned_content = cleaned_content.replace(chunk, "")
        
        # Remove standalone promotional titles
        for title in promotional_titles:
            # Remove as standalone sentence/paragraph
            cleaned_content = re.sub(rf'\b{re.escape(title)}\b\.?\s*', '', cleaned_content)
        
        # Clean up extra whitespace
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        cleaned_content = re.sub(r'\n\s*\n', '\n\n', cleaned_content)
        cleaned_content = cleaned_content.strip()
        
        return cleaned_content
    
    def _clean_title(self, title: str) -> str:
        """Remove specific promotional chunks from content"""
        if not title:
            return ""
        
        # Remove "Demnächst:" prefix
        title = re.sub(r'^Start-ups:\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'^Personalsuche:\s*', '', title, flags=re.IGNORECASE)
        
        return title
    
    def _extract_lead_content(self, soup: BeautifulSoup) -> str:
        """Extract lead paragraphs that trafilatura might miss"""
        lead_selectors = [
            '.article-lead',
        ]
        
        for selector in lead_selectors:
            lead_elem = soup.select_one(selector)
            if lead_elem:
                lead_text = lead_elem.get_text(strip=True)
                if lead_text and len(lead_text) > 50:
                    return lead_text
        
        return ""
    
    def extract_content(self, url: str) -> Optional[Dict]:
        """Extract article content using trafilatura + newspaper3k for images"""
        try:
            import trafilatura
            from newspaper import Article
            from newspaper import Config
            
            logger.info(f"Extracting content from: {url}")
            
            # Fetch the page
            response = self.session.get(url, timeout=PROXY_TIMEOUT)
            response.raise_for_status()
            
            content = trafilatura.extract(response.text, 
                                        include_comments=False,
                                        include_tables=True,
                                        include_formatting=False,
                                        favor_precision=True)
            
            # Extract metadata with trafilatura
            metadata = trafilatura.extract_metadata(response.text)
            
            # Extract images with newspaper3k (better image extraction)
            try:
                config = Config()
                config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                config.fetch_images = True
                
                article = Article(url, config=config)
                article.set_html(response.text)
                article.parse()
                
                # Get images from newspaper3k
                main_image = article.top_image
                all_images = list(article.images) if article.images else []
                
                # Use newspaper3k title/author as fallback if trafilatura didn't get them
                title = metadata.title if metadata and metadata.title else article.title
                author = metadata.author if metadata and metadata.author else ', '.join(article.authors) if article.authors else ''

                # There was a problem with extracting lead paragraph only in cash.ch
                if 'www.cash.ch' in url:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                    lead_content = self._extract_lead_content(soup)
                    
                    # Prepend lead if found and not already in content
                    if lead_content and lead_content not in content:
                        content = lead_content + "\n\n" + content
                
            except Exception as e:
                logger.warning(f"Newspaper3k image extraction failed: {e}")
                main_image = None
                all_images = []
                title = metadata.title if metadata else ''
                author = metadata.author if metadata else ''
            
            if not content or len(content) < 200:
                logger.warning(f"Content too short for {url}: {len(content) if content else 0} chars")
                return None

            # Clean promotional content
            content = self._clean_promotional_content(content=content, title=title)

            sentiment = analyze_sentiment(text=content)
            print(sentiment)

            keywords, topics = self._extract_nlp_features(content=content, title=title)

            # Build article data
            content_data = {
                'id': str(uuid.uuid4()),
                'title': title or '',
                'content': content,
                'description': metadata.description if metadata else '',
                'pub_date': metadata.date if metadata and metadata.date else datetime.now().isoformat(),
                'creator': author,
                'media_url': main_image,  # Best image from newspaper3k
                'language': metadata.language if metadata else 'de',
                'images': all_images,  # All images from newspaper3k
                'article_link': url,
                'extracted_at': datetime.now().isoformat(),
                'sentiment': sentiment,
                'keywords': keywords,
                'topics': topics
            }
            
            logger.info(f"Successfully extracted content: {len(content_data['content'])} chars, {len(all_images)} images, main_image: {bool(main_image)}")
            return content_data

        except ImportError as e:
            logger.error(f"Import error: {e}")
            return None
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return None

    def _extract_nlp_features(self, content: str, title: str) -> Dict:
        """Extract keywords, topics from article content"""
        if not title and not content:
            return None
        input_text = f"{title} {content[:250]}"
        try:
            completion_result = generate_completion_keywords_topics(prompt_input=input_text)
            if completion_result:
                try:
                    result_dict = json.loads(completion_result)
                    if 'keywords' in result_dict and result_dict['keywords'] not in ['not enough information', 'none', '']:
                        keywords = [keyword.strip() for keyword in result_dict['keywords'].split(',')]
                    if 'topics' in result_dict and result_dict['topics'] not in ['none', '']:
                        topics = [topic.strip() for topic in result_dict['topics'].split(',')]
                    return keywords, topics
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing completion result: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Error generation completion result for keywords/topics: {e}")
        return [], []

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title"""
        # Try different title selectors
        selectors = [
            'h1',
            '.entry-title',
            '.post-title', 
            '.article-title',
            'title',
            '.article-header'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if title and len(title) > 10:  # Reasonable title length
                    return title
        
        return ""
    
    def _clean_content(self, content: str) -> str:
        """Clean unwanted text patterns from article content"""
        if not content:
            return ""
        
        # Define patterns to remove
        patterns_to_remove = [
            r"Teilen\s*Merken\s*Drucken\s*Kommentare\s*Google News",
            r"\bTeilen\b",
            r"\bMerken\b", 
            r"\bDrucken\b",
            r"\bKommentare\b",
            r"\bGoogle News\b",
            r"\(AWP\)",
            r"Melden Sie sich an und diskutieren Sie mit\."        ]
        
        end_markers = [
            r"Dieser Artikel erschien zuerst in",
            r"Themen\s*Spezial",
            r"Epaper lesen",
            r"Der HandelDeutscher Fachverlag",
            r"© dfv Mediengruppe",
            r"Mehr lesen\s+Wer Personal sucht",  # Specific promotional text
            r"Mehr lesen\s+Retail Media ist im Trend",  # Another promotional pattern
            # Handle repeated promotional content patterns
            r"Lager,\s*Straße,\s*Lieferkette",
            r"Micro-Influencer verändern das GeschäftDaten gegen Diebe",
        ]
        
        cleaned_content = content
        
        # Remove everything after end markers (keep only content before)
        for marker in end_markers:
            match = re.search(marker, cleaned_content, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                cleaned_content = cleaned_content[:match.start()]
                logger.debug(f"Truncated content at marker: {marker}")
                break  # Only use the first marker found
        
        # Remove specific patterns
        for pattern in patterns_to_remove:
            cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up extra whitespace
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)  # Multiple spaces to single
        cleaned_content = re.sub(r'\n\s*\n', '\n\n', cleaned_content)  # Multiple newlines to double
        cleaned_content = cleaned_content.strip()
        
        return cleaned_content
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main article content"""
        # Try different content selectors first
        content_selectors = [
            '.article-body',  # This specific site uses this
            '.entry-content',
            '.post-content',
            '.article-content',
            '.content',
            'article',
            '.main-content',
            '#content'
        ]
        
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                # Make a copy to work with
                content_copy = content_div.__copy__()
                
                # Remove unwanted elements from the copy
                for element in content_copy(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    element.decompose()
                
                # Remove sharing sections
                for element in content_copy.find_all('section', class_=lambda x: x and 'share' in x.lower()):
                    element.decompose()
                
                # Remove banner ads and iframes
                for element in content_copy.find_all(['div', 'iframe'], class_=lambda x: x and any(
                    keyword in x.lower() for keyword in ['banner', 'ad-', 'advertisement', 'recommendation']
                )):
                    element.decompose()
                
                # Remove recommendation slots specifically
                for element in content_copy.find_all('div', class_=lambda x: x and 'recommendation' in x.lower()):
                    element.decompose()
                
                # Remove partner content sections by finding text content
                for element in content_copy.find_all(string=lambda x: x and 'partner-inhalte' in x.lower()):
                    # Remove the parent container
                    parent = element.parent
                    while parent and parent.name != 'div':
                        parent = parent.parent
                    if parent:
                        # Try to find the recommendation container
                        recommendation_parent = parent
                        for _ in range(5):  # Go up max 5 levels
                            if recommendation_parent and any('recommendation' in cls.lower() for cls in recommendation_parent.get('class', [])):
                                recommendation_parent.decompose()
                                break
                            recommendation_parent = recommendation_parent.parent if recommendation_parent else None
                
                # Remove elements with specific IDs that look like ads
                for element in content_copy.find_all(id=lambda x: x and 'ad' in x.lower()):
                    element.decompose()
                
                # Remove all iframes (usually ads)
                for iframe in content_copy.find_all('iframe'):
                    iframe.decompose()
                
                # Now extract text from paragraphs specifically
                paragraphs = content_copy.find_all('p')
                content_lines = []
                
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:  # Only substantial paragraphs
                        # Skip lines that look like promotional content
                        if any(keyword in text.lower() for keyword in [
                            'facebook', 'twitter', 'linkedin', 'xing', 'whatsapp', 'email',
                            'teilen', 'share', 'artikel teilen', 'diesen artikel', 'publireportage',
                            'partner-inhalte', 'präsentiert von'
                        ]):
                            continue
                        content_lines.append(text)
                
                if content_lines and len('\n\n'.join(content_lines)) > 200:
                    content = '\n\n'.join(content_lines)
                    return self._clean_content(content)
        
        # Fallback: get all paragraphs from body
        paragraphs = soup.find_all('p')
        if paragraphs:
            content = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            return self._clean_content(content)
        
        return ""     
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description"""
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content']
        
        # Try Open Graph description
        og_desc = soup.find('meta', attrs={'property': 'og:description'})
        if og_desc and og_desc.get('content'):
            return og_desc['content']
        
        return ""
    
    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extract article author"""
        # Try different author selectors
        author_selectors = [
            '.author',
            '.byline',
            '.post-author',
            '[rel="author"]'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                author = element.get_text(strip=True)
                if author:
                    return author
        
        # Try meta author
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author and meta_author.get('content'):
            return meta_author['content']
        
        return ""
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract article images"""
        images = []
        
        # Try Open Graph image first
        og_image = soup.find('meta', attrs={'property': 'og:image'})
        if og_image and og_image.get('content'):
            images.append(self._make_absolute_url(og_image['content'], base_url))
        
        # Try to find main content images
        content_area = soup.find('article') or soup.find('.content') or soup
        img_tags = content_area.find_all('img', src=True)
        
        for img in img_tags[:5]:  # Limit to first 5 images
            src = img.get('src')
            if src:
                absolute_url = self._make_absolute_url(src, base_url)
                if absolute_url not in images:
                    images.append(absolute_url)
        
        return images
    
    def _extract_date(self, soup: BeautifulSoup) -> str:
        """Extract publication date"""
        # Try different date selectors
        date_selectors = [
            'time[datetime]',
            '.published',
            '.post-date',
            '.entry-date'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                # Try datetime attribute first
                if element.get('datetime'):
                    return element['datetime']
                # Otherwise get text content
                date_text = element.get_text(strip=True)
                if date_text:
                    try:
                        # Basic date parsing - you might want to improve this
                        return datetime.now().isoformat()  # Fallback for now
                    except:
                        pass
        
        return datetime.now().isoformat()
    
    def _make_absolute_url(self, url: str, base_url: str) -> str:
        """Convert relative URL to absolute"""
        if url.startswith('http'):
            return url
        return urljoin(base_url, url)

class NewsScraperStandalone:
    """Standalone news scraper system"""
    
    def __init__(self, config_path: str = 'config.json'):
        self.config_manager = ConfigManager(config_path)
        
        # Create session with proxy if available
        self.session = requests.Session()
        proxy_ip = os.getenv('STORM_PROXY_IP')
        if proxy_ip:
            proxy_dict = {
                'http': f'http://{proxy_ip}',
                'https': f'http://{proxy_ip}'
            }
            self.session.proxies.update(proxy_dict)
            logger.info(f"Using proxy: {proxy_ip}")
        
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.content_extractor = ContentExtractor(self.session)
        self.stats = {}

    
    def scrape_source(self, source_name: str, max_articles: int = None) -> Dict:
        """Scrape a single news source"""
        logger.info(f"Starting scrape for source: {source_name}")
        
        source_config = self.config_manager.get_source_by_name(source_name)
        if not source_config:
            logger.error(f"Source {source_name} not found in config")
            return {'success': False, 'error': 'Source not found'}
        
        try:
            # Get sitemap handler
            sitemap_type = source_config.get('sitemap_type', 'standard')
            handler = SitemapHandlerFactory.create_handler(sitemap_type, self.session)
            
            # Get article URLs
            article_urls = handler.get_article_urls(source_config)
            
            if not article_urls:
                logger.warning(f"No articles found for source {source_name}")
                return {'success': False, 'error': 'No articles found'}
            
            # Limit articles if specified
            if max_articles:
                article_urls = article_urls[:max_articles]
            
            # Extract content from each article
            extracted_articles = []
            failed_articles = []
            
            for i, (url, lastmod) in enumerate(article_urls):
                logger.info(f"Processing article {i+1}/{len(article_urls)}")
                
                try:
                    article_data = self.content_extractor.extract_content(url)
                    
                    if article_data:
                        # Add source information
                        article_data.update({
                            'source_title': source_config.get('source_title', ''),
                            'source_link': source_config.get('source_url', ''),
                            'source_country': source_config.get('source_country', ''),
                            'language': source_config.get('language', 'de'),
                            'lastmod': lastmod
                        })
                        
                        extracted_articles.append(article_data)
                        
                        if TEST_MODE:
                            print(f"✅ Extracted: {article_data['title']}")
                            print(f"📝 Content ({len(article_data['content'])} chars):")
                            print("-" * 80)
                            print(article_data['content'])
                            print("-" * 80)
                            print(f"📸 Images: {len(article_data['images'])}")
                            if article_data['images']:
                                for i, img in enumerate(article_data['images'][:3]):
                                    print(f"   {i+1}. {img}")
                            print(f"👤 Author: {article_data['creator']}")
                            print(f"📅 Date: {article_data['pub_date']}")
                            print(f"🔗 URL: {article_data['article_link']}")
                            print("\n" + "="*100 + "\n")
                        else:
                            # In production, you would save to database here
                            self._save_article_to_database(article_data)
                        
                    else:
                        failed_articles.append(url)
                        logger.warning(f"Failed to extract content from: {url}")
                    
                    # Add delay between requests
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing article {url}: {e}")
                    failed_articles.append(url)
            
            # Build result
            result = {
                'success': True,
                'source_name': source_name,
                'source_title': source_config.get('source_title', ''),
                'sitemap_type': sitemap_type,
                'total_urls_found': len(article_urls),
                'articles_extracted': len(extracted_articles),
                'articles_failed': len(failed_articles),
                'success_rate': len(extracted_articles) / len(article_urls) if article_urls else 0,
                'extracted_articles': extracted_articles,
                'failed_urls': failed_articles
            }
            
            logger.info(f"Completed scraping {source_name}: {len(extracted_articles)}/{len(article_urls)} articles extracted")
            return result
            
        except Exception as e:
            logger.error(f"Error scraping source {source_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def scrape_all_sources(self, shard_color: str = 'red', max_articles_per_source: int = None) -> Dict:
        """Scrape all sources in a shard"""
        sources = self.config_manager.get_sources_by_shard(shard_color)
        logger.info(f"Found {len(sources)} sources for shard {shard_color}")
        
        results = {}
        total_extracted = 0
        total_failed = 0
        
        for source in sources:
            source_name = source['name']
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing source: {source_name}")
            logger.info(f"{'='*60}")
            
            result = self.scrape_source(source_name, max_articles_per_source)
            results[source_name] = result
            
            if result['success']:
                total_extracted += result['articles_extracted']
                total_failed += result['articles_failed']
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SCRAPING COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Sources processed: {len(sources)}")
        logger.info(f"Total articles extracted: {total_extracted}")
        logger.info(f"Total articles failed: {total_failed}")
        
        return {
            'total_sources': len(sources),
            'total_extracted': total_extracted,
            'total_failed': total_failed,
            'source_results': results
        }
    
    def _save_article_to_database(self, article_data: Dict):
        """Save article to database using existing database structure"""
        if TEST_MODE:
            logger.info(f"TEST MODE: Would save article {article_data['id']} to database")
            return
        
        try:
            from db.db_manager import DatabaseManager
            from db.models import NewsArticle
            from uuid import uuid4
            from datetime import datetime
            from sqlalchemy.exc import IntegrityError
            
            # Initialize database connection
            db_manager = DatabaseManager()
            db_manager.pg_manager.init_session()
            session = db_manager.pg_manager.session
            
            # Parse publication date
            try:
                if article_data.get('pub_date'):
                    # Handle different date formats
                    pub_date_str = article_data['pub_date']
                    if 'T' in pub_date_str:
                        pub_date = datetime.fromisoformat(pub_date_str.replace('Z', ''))
                    else:
                        pub_date = datetime.strptime(pub_date_str, '%Y-%m-%d %H:%M:%S')
                else:
                    pub_date = datetime.now()
            except (ValueError, TypeError):
                pub_date = datetime.now()
                logger.warning(f"Could not parse pub_date, using current time: {article_data.get('pub_date')}")
            
            # Create NewsArticle instance
            news_article = NewsArticle(
                crawl_id=None,  # You may want to add this to your article_data
                guid=article_data['id'],
                category=None,  # Add if you have category mapping
                pub_date=pub_date,
                link=article_data['article_link'],
                title=article_data['title'],
                creator=article_data.get('creator'),
                description=article_data.get('description'),
                channel_title=article_data.get('source_title'),
                channel_link=article_data.get('source_link'),
                language=article_data.get('language', 'de'),
                source=None,  # Add source type mapping if needed
                media_url=article_data.get('media_url'),
                media_type="image/jpeg" if article_data.get('media_url') else None,
                media_title=None,
                media_description=article_data.get('media_description'),
                media_credit=None,
                media_thumbnail=None,
                content=article_data['content'],
                keywords=article_data.get('keywords', []),
                topics=article_data.get('topics', []),
                sentiment_pos=article_data.get('sentiment', {}).get('pos'),
                sentiment_neg=article_data.get('sentiment', {}).get('neg'),
                sentiment_neu=article_data.get('sentiment', {}).get('neu'),
                fetch_attempted=True,
                fetch_date=datetime.now(),
                shard_color=article_data.get('shard_color'),  # Add to your article_data
                news_source_id=article_data.get('news_source_id'),  # Add to your article_data
                processed=False,
            )
            
            # Save to database with duplicate handling
            try:
                session.add(news_article)
                session.commit()
                logger.info(f"Successfully saved article to database: {article_data['title']}")
                
            except IntegrityError as e:
                session.rollback()
                if 'unique_link' in str(e):
                    logger.warning(f"Article with URL already exists: {article_data['article_link']}")
                else:
                    logger.error(f"Database integrity error: {e}")
                    
            except Exception as e:
                session.rollback()
                logger.error(f"Error saving article to database: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Failed to save article {article_data.get('id', 'unknown')}: {e}")
            raise
        finally:
            if 'session' in locals():
                session.close()

def main():
    """Main function for standalone usage"""
    print("\n🚀 Standalone News Scraper")
    print("="*50)
    
    if TEST_MODE:
        print("🧪 Running in TEST MODE - no database writes")
    
    # Initialize scraper
    scraper = NewsScraperStandalone()
    
    # Interactive mode
    print("\nChoose an option:")
    print("1. Scrape a specific source")
    print("2. Scrape all sources")
    print("3. Test sitemap only (no content extraction)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        source_name = input("Enter source name: ").strip()
        max_articles = input("Max articles (press Enter for all): ").strip()
        max_articles = int(max_articles) if max_articles else None
        
        result = scraper.scrape_source(source_name, max_articles)
        
        if result['success']:
            print(f"\n✅ Success! Extracted {result['articles_extracted']} articles")
        else:
            print(f"\n❌ Failed: {result['error']}")
    
    elif choice == '2':
        shard_color = input("Enter shard color (default: red): ").strip() or 'red'
        max_articles = input("Max articles per source (press Enter for all): ").strip()
        max_articles = int(max_articles) if max_articles else None
        
        results = scraper.scrape_all_sources(shard_color, max_articles)
        print(f"\n✅ Completed! Total extracted: {results['total_extracted']} articles")
    
    elif choice == '3':
        # Test sitemap only
        source_name = input("Enter source name: ").strip()
        source_config = scraper.config_manager.get_source_by_name(source_name)
        
        if source_config:
            sitemap_type = source_config.get('sitemap_type', 'standard')
            handler = SitemapHandlerFactory.create_handler(sitemap_type, scraper.session)
            urls = handler.get_article_urls(source_config)
            
            print(f"\n✅ Found {len(urls)} URLs in sitemap")
            for i, (url, lastmod) in enumerate(urls[:5]):  # Show first 5
                print(f"{i+1}. {url}")
        else:
            print(f"❌ Source {source_name} not found")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()