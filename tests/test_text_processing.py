import pytest
from src.text_processing import clean_text, extract_skills, looks_like_resume_or_jd

def test_clean_text():
    assert clean_text('Python/Java (C++)') == 'python/java c++'
    assert clean_text('  hello   world  ') == 'hello world'
    assert clean_text('') == ''
    assert clean_text(None) == ''

def test_extract_skills():
    text = 'I know Python, Java, and C++.'
    skills = extract_skills(text)
    assert 'python' in skills
    assert 'java' in skills
    assert 'c++' in skills
    assert 'ruby' not in skills

    # Check boundaries
    text2 = 'pythonic java.net'
    skills2 = extract_skills(text2)
    assert 'python' not in skills2
    assert 'java' in skills2

def test_looks_like_resume_or_jd():
    assert looks_like_resume_or_jd('word ' * 30) is True
    assert looks_like_resume_or_jd('word ' * 29) is False
    assert looks_like_resume_or_jd('') is False
    assert looks_like_resume_or_jd(None) is False
