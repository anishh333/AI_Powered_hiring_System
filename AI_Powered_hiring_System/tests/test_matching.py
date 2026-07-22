import pytest
from src.matching import SemanticMatcher, evaluate_match

def test_semantic_matcher():
    corpus = ["I am a software engineer proficient in python and java",
              "Data scientist with experience in machine learning and python",
              "Frontend developer building react apps"]
    matcher = SemanticMatcher(corpus)
    score1 = matcher.similarity("I know python and java", "Looking for java and python engineer")
    assert 0.0 <= score1 <= 1.0

def test_evaluate_match():
    corpus = ["I am a software engineer proficient in python and java"]
    matcher = SemanticMatcher(corpus)
    resume = "I have 5 years of python and react experience."
    jd = "Looking for python and django developer."
    result = evaluate_match(matcher, resume, jd)
    assert result["skill_score"] > 0
    assert "python" in result["matching_skills"]
    assert "django" in result["missing_skills"]
    assert "react" in result["extra_skills"]

def test_evaluate_match_empty_skills():
    corpus = ["I am a software engineer proficient in python and java"]
    matcher = SemanticMatcher(corpus)
    resume = "Empty resume with no known words"
    jd = "Empty job description with no known words"
    result = evaluate_match(matcher, resume, jd)
    assert "final_score" in result
