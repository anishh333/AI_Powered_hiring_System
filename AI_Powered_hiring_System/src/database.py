"""
Database setup using SQLModel, connecting to the local SQLite DB.
"""
import os
from typing import Generator, Optional
from sqlmodel import SQLModel, Field, create_engine, Session

# Database path definition
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "resume_database.sqlite")
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)


class JobDescription(SQLModel, table=True):
    """Stores parsed job descriptions."""
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    domain: str
    raw_text: str
    skills: str  # Comma-separated list of skills


class Candidate(SQLModel, table=True):
    """Stores parsed candidates uploaded via the UI."""
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    domain: str
    years_of_experience: int
    education: str
    skills: str  # Comma-separated list of skills
    summary: str
    raw_text: str


def init_db():
    """Initializes tables in the SQLite database."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """Provides a transactional database session."""
    with Session(engine) as session:
        yield session
