from __future__ import annotations

from contextlib import contextmanager
from typing import Mapping, Any, Optional
from dotenv import load_dotenv
import os
from psycopg2 import OperationalError
import time
from sqlalchemy import create_engine, engine, QueuePool
from sqlalchemy.orm import Session, sessionmaker
import logging

load_dotenv()

loki_logger = logging.getLogger('web_crawler')


def connect_to_db(db_url: str) -> engine.Engine:
    retries = 0
    max_retries = 5
    retry_interval = 5
    while retries < max_retries:
        try:
            time.sleep(5)
            db_engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            connection = db_engine.connect()
            connection.close()

            # loki_logger.info("Database connection successful")
            return db_engine
        except OperationalError as e:
            retries += 1

            loki_logger.error("Database connection attempt failed, retrying...")
            time.sleep(retry_interval)

    raise Exception("Max retries reached. Could not connect to the database.")


class PostgresDBManager:
    def __init__(self) -> None:
        self._pg_db_details: dict[str, Optional[str]] = {
            "dbname": os.getenv('PG_DB_NAME'),
            "user": os.getenv('PG_USER'),
            "password": os.getenv('PG_PASSWORD'),
            "host": os.getenv('PG_HOST'),
            "port": os.getenv('PG_PORT')
        }

        database_url = f"postgresql://{self._pg_db_details['user']}:{self._pg_db_details['password']}@{self._pg_db_details['host']}:{self._pg_db_details['port']}/{self._pg_db_details['dbname']}"

        # loki_logger.info(f"Connecting to: postgresql://{self._pg_db_details['user']}:******@{self._pg_db_details['host']}:{self._pg_db_details['port']}/{self._pg_db_details['dbname']}")

        self._engine: engine.Engine = connect_to_db(database_url)
        self._sessionmaker: sessionmaker = sessionmaker(bind=self._engine)
        self._session: Optional[Session] = None

    def init_session(self) -> None:
        if self._session is None or not self._session.is_active:
            self._session = self._sessionmaker(expire_on_commit=False)

    @property
    def session(self) -> Session:
        self.init_session()
        assert self._session is not None
        return self._session

    def commit(self) -> None:
        if self._session is not None:
            self._session.commit()

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None

    def is_connected(self):  # type: ignore
        return self._session is not None and self._session.is_active

    @contextmanager
    def session_scope(self):  # type: ignore
        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

