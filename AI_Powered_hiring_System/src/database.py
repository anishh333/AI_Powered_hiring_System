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
    raw_pdf: Optional[bytes] = Field(default=None)


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
    raw_pdf: Optional[bytes] = Field(default=None)


def init_db():
    """Initializes tables in the SQLite database and runs migrations if needed."""
    SQLModel.metadata.create_all(engine)
    
    # Check if raw_pdf columns exist and add them if they don't
    import sqlite3
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check jobdescription table
        cursor.execute("PRAGMA table_info(jobdescription)")
        cols_jd = [row[1] for row in cursor.fetchall()]
        if cols_jd and "raw_pdf" not in cols_jd:
            cursor.execute("ALTER TABLE jobdescription ADD COLUMN raw_pdf BLOB")
            conn.commit()
            
        # Check candidate table
        cursor.execute("PRAGMA table_info(candidate)")
        cols_cand = [row[1] for row in cursor.fetchall()]
        if cols_cand and "raw_pdf" not in cols_cand:
            cursor.execute("ALTER TABLE candidate ADD COLUMN raw_pdf BLOB")
            conn.commit()
            
        conn.close()
    except Exception as e:
        print(f"Database migration error: {e}")


def get_session() -> Generator[Session, None, None]:
    """Provides a transactional database session."""
    with Session(engine) as session:
        yield session
