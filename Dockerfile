FROM python:3.9-slim

# Install Python packages
RUN pip install lxml_html_clean lxml nltk openai python-dotenv psycopg2-binary SQLAlchemy elasticsearch==8.12.1
RUN pip install trafilatura newspaper3k==0.2.8 requests beautifulsoup4

WORKDIR /app
COPY .env .
COPY standalone_scraper.py .
COPY config.json .
COPY db/ ./db/

CMD ["python", "standalone_scraper.py"]