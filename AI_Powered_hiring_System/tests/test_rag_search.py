import pytest
from src.rag_search import ResumeSearchEngine

def test_resume_search_engine(tmp_path):
    import sqlite3
    db_path = tmp_path / "resume_database.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE resumes (
            ResumeID INTEGER PRIMARY KEY,
            Category TEXT,
            Name TEXT,
            Skills TEXT,
            Summary TEXT,
            Experience TEXT,
            Text TEXT
        )
    """)
    conn.execute("""
        INSERT INTO resumes (Category, Name, Skills, Summary, Experience, Text)
        VALUES ('Engineering', 'John Doe', 'Python, Java', 'Backend Dev', '5 years', 'I am a python and java engineer')
    """)
    conn.commit()
    conn.close()

    engine = ResumeSearchEngine(str(db_path))
    assert engine.is_relevant("Looking for python developer") is True
    assert engine.is_relevant("What is the recipe for cake?") is False
    
    results = engine.search("python")
    assert not results.empty
    assert "John Doe" in results["Name"].values
    
    common_skills = engine.common_skills_for("python")
    assert "python" in common_skills
    assert "java" in common_skills

    answer = engine.answer("Looking for a python developer")
    assert answer["relevant"] is True
    assert "John Doe" in answer["message"]

    # Off topic answer
    answer_off = engine.answer("How to bake a cake?")
    assert answer_off["relevant"] is False
