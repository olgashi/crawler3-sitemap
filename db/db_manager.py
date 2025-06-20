
from db.postgres_manager import PostgresDBManager
import logging

loki_logger = logging.getLogger('web_crawler')

class DatabaseManager:
    def __init__(self):  # type: ignore
        self._pg_manager = None

    @property
    def pg_manager(self):  # type: ignore
        if not self._pg_manager or not self._pg_manager.is_connected():
            try:
                self._pg_manager = PostgresDBManager()
                self._pg_manager.init_session()
            except Exception as e:
                loki_logger.error(f"Failed to initialize PostgreSQL connection: {str(e)}")
                raise
        return self._pg_manager


    def initialize_connections(self):  # type: ignore
        loki_logger.info("Initializing database connections")
        _ = self.pg_manager

    def close_connections(self):  # type: ignore
        loki_logger.info("Closing database connections")
        if self._pg_manager:
            self._pg_manager.close()

db_manager = DatabaseManager()