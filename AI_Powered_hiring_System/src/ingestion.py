"""
Ingestion module for parsing job descriptions and candidate resumes.
"""
from sqlmodel import Session
import logfire

from .database import JobDescription, Candidate
from .pdf_utils import extract_text_from_pdf
from .text_processing import (
    extract_skills,
    extract_years_of_experience,
    extract_education_level,
    infer_domain,
    extract_summary
)


def process_jd(file_bytes: bytes, filename: str, session: Session) -> int:
    """Extracts, parses, and saves a job description to the database."""
    with logfire.span("Process Job Description: {filename}", filename=filename) as span:
        jd_text = extract_text_from_pdf(file_bytes)
        skills_set = extract_skills(jd_text)
        domain = infer_domain(skills_set)
        
        # Enrich trace with metadata attributes
        span.set_attribute("domain", domain)
        span.set_attribute("skills_count", len(skills_set))
        span.set_attribute("text_length", len(jd_text))
        
        jd = JobDescription(
            filename=filename,
            domain=domain,
            raw_text=jd_text,
            skills=",".join(sorted(skills_set)),
            raw_pdf=file_bytes
        )
        session.add(jd)
        session.commit()
        session.refresh(jd)
        logfire.info("Ingested JD: {filename} with ID: {jd_id}", filename=filename, jd_id=jd.id)
        return jd.id


def process_resume(file_bytes: bytes, filename: str, session: Session) -> Candidate:
    """Extracts, parses, and saves a candidate resume to the database."""
    with logfire.span("Process Resume: {filename}", filename=filename) as span:
        resume_text = extract_text_from_pdf(file_bytes)
        skills_set = extract_skills(resume_text)
        domain = infer_domain(skills_set)
        years_exp = extract_years_of_experience(resume_text)
        edu = extract_education_level(resume_text)
        summary = extract_summary(resume_text)
        
        # Enrich trace with metadata attributes
        span.set_attribute("domain", domain)
        span.set_attribute("skills_count", len(skills_set))
        span.set_attribute("years_experience", years_exp)
        span.set_attribute("education_level", edu)
        span.set_attribute("text_length", len(resume_text))
        
        cand = Candidate(
            filename=filename,
            domain=domain,
            years_of_experience=years_exp,
            education=edu,
            skills=",".join(sorted(skills_set)),
            summary=summary,
            raw_text=resume_text,
            raw_pdf=file_bytes
        )
        session.add(cand)
        session.commit()
        session.refresh(cand)
        logfire.info("Ingested Candidate: {filename} with ID: {cand_id}", filename=filename, cand_id=cand.id)
        return cand
