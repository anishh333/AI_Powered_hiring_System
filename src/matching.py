"""
Semantic similarity between a job description and a resume.

Deliberately avoids Hugging Face / sentence-transformers. Instead this uses a
TF-IDF vector space model (scikit-learn) fitted on the local resume corpus so
that word importance (IDF) reflects real hiring-domain language. This is a
fully local, fully offline "ML" similarity model — nothing is downloaded.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_processing import clean_text, extract_skills


class SemanticMatcher:
    """Fits a TF-IDF space on a background corpus, then scores JD vs resume."""

    def __init__(self, background_corpus: list[str]):
        corpus = [clean_text(t) for t in background_corpus if t and t.strip()]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=6000,
            min_df=2,
        )
        # Fit on the corpus so IDF weights reflect real resume/job language.
        self.vectorizer.fit(corpus if corpus else ["placeholder text"])

    def similarity(self, text_a: str, text_b: str) -> float:
        vecs = self.vectorizer.transform([clean_text(text_a), clean_text(text_b)])
        score = cosine_similarity(vecs[0], vecs[1])[0][0]
        return float(np.clip(score, 0.0, 1.0))


def evaluate_match(matcher: SemanticMatcher, resume_text: str, jd_text: str) -> dict:
    """
    Combine TF-IDF semantic similarity with explicit skill overlap into one
    explainable match score, plus the matching/missing skill lists.
    """
    semantic_score = matcher.similarity(resume_text, jd_text)

    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)

    matching = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)

    if jd_skills:
        skill_score = len(matching) / len(jd_skills)
    else:
        skill_score = semantic_score

    # Weighted blend: skill overlap is the more explainable/reliable signal,
    # semantic similarity captures phrasing/context beyond the fixed vocabulary.
    final_score = 0.6 * skill_score + 0.4 * semantic_score

    return {
        "final_score": round(final_score, 4),
        "semantic_score": round(semantic_score, 4),
        "skill_score": round(skill_score, 4),
        "matching_skills": matching,
        "missing_skills": missing,
        "extra_skills": extra,
        "jd_skills": sorted(jd_skills),
        "resume_skills": sorted(resume_skills),
    }
