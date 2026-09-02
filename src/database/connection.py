import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import Generator

logger = logging.getLogger(__name__)

# Base model for declarative class definitions
Base = declarative_base()

def get_database_url() -> str:
    """
    Returns the database URL from environment or defaults to local SQLite database.
    """
    db_url = os.getenv("DATABASE_URL", "sqlite:///./data/portfolio_research.db")
    if db_url.startswith("sqlite"):
        os.makedirs("data", exist_ok=True)
    return db_url

def get_engine():
    """
    Creates and returns the SQLAlchemy database engine with connection pooling.
    """
    db_url = get_database_url()
    connect_args = {"check_same_thread": False} if "sqlite" in db_url else {}
    engine = create_engine(
        db_url,
        echo=False,
        connect_args=connect_args,
        pool_pre_ping=True
    )
    return engine

# Session factory
engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator:
    """
    FastAPI dependency and context manager for database sessions.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db(target_engine=None):
    """
    Creates all relational tables defined in the schema.
    """
    eng = target_engine or engine
    logger.info(f"Initializing database tables on: {eng.url}")
    Base.metadata.create_all(bind=eng)
    logger.info("Database schema creation successful.")

def reset_db(target_engine=None):
    """
    Drops and recreates all tables.
    """
    eng = target_engine or engine
    logger.warning(f"Resetting database tables on: {eng.url}")
    Base.metadata.drop_all(bind=eng)
    Base.metadata.create_all(bind=eng)
