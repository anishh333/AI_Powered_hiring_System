import os
import pytest
from sqlmodel import Session, select, SQLModel, create_engine
from src.database import JobDescription, Candidate
from src.ingestion import process_jd, process_resume

def test_database_raw_pdf_storage(tmp_path):
    # Setup temporary test database
    db_path = tmp_path / "test_db.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    
    # Initialize SQLModel metadata on the new database engine
    SQLModel.metadata.create_all(engine)
    
    # Mock data
    jd_pdf_bytes = b"%PDF-1.4 Mock JD PDF content"
    resume_pdf_bytes = b"%PDF-1.4 Mock Resume PDF content"
    
    # Process JD and Candidate using a Session tied to the temporary test DB
    with Session(engine) as session:
        # Ingestion logic normally parses text from PDF, but since extract_text_from_pdf returns empty string on mock PDF
        # without crashing, this is perfect for checking database insertion and retrieval of raw_pdf column.
        jd_id = process_jd(jd_pdf_bytes, "test_jd.pdf", session)
        cand = process_resume(resume_pdf_bytes, "test_resume.pdf", session)
        
        # Verify saved job description
        jd = session.get(JobDescription, jd_id)
        assert jd is not None
        assert jd.filename == "test_jd.pdf"
        assert jd.raw_pdf == jd_pdf_bytes
        
        # Verify saved candidate
        candidate = session.get(Candidate, cand.id)
        assert candidate is not None
        assert candidate.filename == "test_resume.pdf"
        assert candidate.raw_pdf == resume_pdf_bytes
