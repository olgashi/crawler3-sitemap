# Crawler3 Sitemap - News Scraper

A standalone Python application that crawls German/Austrian trade publication websites by discovering articles through their sitemaps and extracting structured content for storage in PostgreSQL and Elasticsearch.

## How It Works

### Architecture Overview

This application is a **sitemap-based web scraper** that:

1. **Reads configuration** from `config.json` containing news sources
2. **Discovers articles** through various sitemap strategies
3. **Extracts content** using BeautifulSoup and custom selectors
4. **Enriches data** with AI-powered keyword/topic extraction and sentiment analysis
5. **Stores results** in PostgreSQL database and Elasticsearch index

### Sitemap Strategies

The application supports multiple sitemap discovery methods:

#### 1. Standard Sitemaps (`sitemap_type: "standard"`)
Fetches standard XML sitemaps directly:
```json
{
  "sitemap_type": "standard",
  "sitemap_urls": ["https://example.com/sitemap.xml"]
}
```

#### 2. Date-Dynamic Sitemaps (`sitemap_type: "date_dynamic"`)
Generates sitemap URLs with date patterns for sites that organize sitemaps by month:
```json
{
  "sitemap_type": "date_dynamic", 
  "sitemap_pattern": "https://example.com/sitemap.xml?f={YYYY_MM}",
  "fallback_months": 3
}
```

#### 3. JetPack Table Sitemaps (`sitemap_type: "jetpack_table"`)
For WordPress sites using JetPack sitemap format:
```json
{
  "sitemap_type": "jetpack_table",
  "sitemap_urls": ["https://example.com/sitemap-1.xml"]
}
```

#### 4. Category Pages (`sitemap_type: "category_pages"`)
Scrapes article links from category listing pages:
```json
{
  "sitemap_type": "category_pages",
  "category_urls": ["https://example.com/category/news/"],
  "max_pages": 5
}
```

### Proxy Usage

**Yes, the application uses proxies:**

- Configured via `STORM_PROXY_IP` environment variable (`5.79.73.131:13010`)
- Proxy is attempted first for all HTTP requests
- **Automatic fallback** to direct connection if proxy fails
- 60-second timeout for proxy requests (`PROXY_TIMEOUT = 60`)

The proxy implementation in `standalone_scraper.py:252-271` provides robust fallback:
```python
def _fetch_with_fallback(self, url: str, timeout: int = 30) -> requests.Response:
    try:
        # Try with proxy first
        response = self.session.get(url, timeout=timeout)
        return response
    except (requests.exceptions.SSLError, requests.exceptions.ProxyError, 
            requests.exceptions.ConnectionError) as e:
        # Create session without proxy and retry
        fallback_session = requests.Session()
        response = fallback_session.get(url, timeout=timeout)
        return response
```

### Content Processing

#### Article Extraction
- **Multi-selector approach**: Tries various CSS selectors for title/content extraction
- **Content cleaning**: Removes promotional text, navigation elements, and ads
- **Image extraction**: Finds and processes article images
- **Date parsing**: Handles multiple date formats from different sources

#### AI Enhancement
- **OpenAI GPT-4 integration**: Extracts up to 5 keywords and 5 topics per article
- **Sentiment analysis**: Uses NLTK's VADER sentiment analyzer
- **Language detection**: Supports German content primarily

#### Deduplication
- **Elasticsearch checking**: Queries existing articles before processing new URLs
- **PostgreSQL constraints**: Unique constraint on article URLs prevents duplicates

## Adding New Sites

### Can you just add site links and expect it to work?

**Partially yes, but configuration is required:**

1. **Add to config.json**: Sites must be properly configured with the correct sitemap strategy
2. **Choose correct sitemap_type**: The application needs to know how to discover articles
3. **Test first**: Use `TEST_MODE = True` to verify extraction works
4. **Content selectors may need adjustment**: Some sites may require custom CSS selectors

### Adding a New Source

1. **Identify the sitemap strategy** by examining the target site:
   ```bash
   curl https://example.com/sitemap.xml
   curl https://example.com/robots.txt  # Often lists sitemaps
   ```

2. **Add configuration to config.json**:
   ```json
   {
     "name": "example_site",
     "source_title": "Example News",
     "source_url": "https://example.com",
     "sitemap_type": "standard",
     "sitemap_urls": ["https://example.com/sitemap.xml"],
     "language": "de",
     "shard_color": "red", 
     "enabled": true,
     "id": "example-site",
     "source_country": "DE"
   }
   ```

3. **Test the configuration**:
   ```bash
   python standalone_scraper.py
   ```
   Check logs for successful URL discovery and content extraction.

4. **Adjust if needed**: If content extraction fails, you may need to modify the content selectors in the `_extract_main_content()` method.

### Success Rate Expectations

Most sites work well with minimal configuration if they have standard sitemaps. The application includes fallback mechanisms and handles common edge cases, making it quite robust for adding new German/Austrian news sources.

## Environment Setup

### Required Environment Variables (.env)
```
PG_DB_NAME=your_db_name
PG_USER=your_db_user  
PG_PASSWORD=your_db_password
PG_HOST=your_db_host
PG_PORT=5432
OPENAI_API_KEY=your_openai_key
STORM_PROXY_IP=your_proxy_ip:port  # Optional but recommended
ELASTICSEARCH_LB_IP=your_elasticsearch_ip
SHARD_COLOR=red  # For distributed processing
```

### Installation
```bash
pip install requests beautifulsoup4 openai elasticsearch nltk python-dotenv sqlalchemy psycopg2-binary
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Operation Modes

- **TEST_MODE = True**: Prints extracted content without database writes
- **TEST_MODE = False**: Full production mode with database persistence  
- **HISTORICAL_MODE**: For initial bulk scraping vs. incremental updates

## Database Schema

Articles are stored in PostgreSQL with the following key fields:
- **Metadata**: title, creator, pub_date, link, source info
- **Content**: full article text, description, images
- **AI Fields**: keywords[], topics[], sentiment scores
- **Processing**: fetch_date, processed status, shard_color


### The following sources have been deleted from crawl_target table and now are scraped via this application:
- https://www.libertynation.com
- https://thefederalist.com
- https://www.phillytrib.com
- https://radaronline.com
- https://www.freepressjournal.in
- https://wgntv.com
- https://newsindiatimes.com
- https://fox5sandiego.com
- https://www.newsnationnow.com
- https://www.pagina12.com.ar