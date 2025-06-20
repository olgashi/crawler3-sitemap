from datetime import datetime

from sqlalchemy import Column, Integer, Text, Boolean, TIMESTAMP, ForeignKey, UUID, ARRAY, Numeric, UniqueConstraint, insert
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
import logging
from typing import TypeVar, Optional, Sequence, Any
from enum import Enum
from sqlalchemy.dialects.postgresql import ENUM


Base = declarative_base()
T = TypeVar('T', bound=Base)  # type: ignore
loki_logger = logging.getLogger('web_crawler')


class SourceType(Enum):
    rss = "rss"


class CrawlStatus(Enum):
    PENDING = 'pending'
    QUEUED = 'queued'
    COMPLETED = 'completed'
    FAILED = 'failed'


class DeadCrawlStatus(Enum):
    BAD_DATA = 'bad_data'
    NO_ITEMS = 'no_items'
    NO_CONTENT = 'no_content'
    GOOD = 'good'


class NewsArticle(Base):  # type: ignore
    __tablename__ = 'article'

    id = Column(Integer, primary_key=True, autoincrement=True)
    crawl_id = Column(UUID(as_uuid=True))
    category = Column(Text)
    guid = Column(Text)
    pub_date = Column(TIMESTAMP)
    link = Column(Text)
    title = Column(Text)
    creator = Column(Text)
    description = Column(Text)
    channel_title = Column(Text)
    channel_link = Column(Text)
    language = Column(Text)
    source = Column(ENUM(SourceType, name='source_type', create_type=False))  # type: ignore
    media_url = Column(Text)
    media_type = Column(Text)
    media_title = Column(Text)
    media_description = Column(Text)
    media_credit = Column(Text)
    media_thumbnail = Column(Text)
    content = Column(Text)
    keywords = Column(ARRAY(Text))  # type: ignore
    topics = Column(ARRAY(Text))  # type: ignore
    fetch_attempted = Column(Boolean, default=False)
    processed = Column(Boolean, default=False)
    fetch_date = Column(TIMESTAMP)
    created_at = Column(TIMESTAMP, default=func.now())
    sentiment_pos = Column(Numeric)
    sentiment_neg = Column(Numeric)
    sentiment_neu = Column(Numeric)
    shard_color = Column(Text)
    news_source_id = Column(Numeric)
    last_modified = Column(TIMESTAMP, default=func.now())
    processed = Column(Boolean, default=False)
    
    __table_args__ = (
        UniqueConstraint('link', name='unique_link'),
    )

class NewsSource(Base):  # type: ignore
    __tablename__ = 'news_source'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_country = Column(Text)
    language = Column(Text)
    region = Column(Text)
    source_title = Column(Text)
    source_link = Column(Text, nullable=False)
    media_type = Column(Text)
    category = Column(Text)
    subcategory = Column(Text)
    rss_link = Column(Text, unique=True, nullable=False)
    source_description = Column(Text)


