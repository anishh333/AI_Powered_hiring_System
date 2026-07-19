"""
Retrieval-augmented search over the local resume database.

Retrieval uses a TF-IDF vector space (scikit-learn), not a Hugging Face
embedding model. A lightweight relevance guardrail rejects queries that are
unrelated to resumes/hiring, so the system doesn't try to "hallucinate" an
answer to an out-of-scope question.
"""
import sqlite3
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_processing import clean_text, extract_skills

# Extra domain words (beyond what's already in the corpus vocabulary) that
# signal a query is actually about hiring/resumes, used by the guardrail.
DOMAIN_HINTS = {
    "resume", "resumes", "cv", "candidate", "candidates", "applicant",
    "hire", "hiring", "recruit", "recruiter", "job", "role", "position",
    "developer", "engineer", "experience", "skill", "skills", "years",
    "qualified", "background", "profile", "team", "manager", "analyst",
}

# Below this combined relevance score, the query is treated as out of scope.
RELEVANCE_THRESHOLD = 0.18

# Common general-knowledge / small-talk topics that should never be answered
# by this assistant, even if they happen to share a stray word with the
# resume corpus (e.g. "president" appears in job titles like "Vice President").
OFF_TOPIC_WORDS = {
    "president", "capital", "weather", "recipe", "poem", "joke", "movie",
    "song", "lyrics", "planet", "war", "election", "prime", "minister",
    "sports", "football", "cricket", "score", "temperature", "forecast",
    "translate", "horoscope", "celebrity", "religion", "politics",
}
# A query also needs at least this many content words recognised by the
# corpus vocabulary before a similarity score alone is trusted (this stops a
# single coincidental word match, e.g. "random", from passing the guardrail).
MIN_RECOGNISED_WORDS = 2


class ResumeSearchEngine:
    def __init__(self, db_path: str):
        conn = sqlite3.connect(db_path)
        self.df = pd.read_sql_query(
            "SELECT ResumeID, Category, Name, Skills, Summary, Experience, Text "
            "FROM resumes", conn,
        )
        conn.close()

        self._clean_texts = [clean_text(t) for t in self.df["Text"].fillna("")]
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=8000, min_df=2,
        )
        self.doc_matrix = self.vectorizer.fit_transform(self._clean_texts)

        self.categories = sorted(self.df["Category"].dropna().unique().tolist())
        self._category_vocab = {c.lower() for c in self.categories}

    # ---- guardrail ------------------------------------------------------
    def is_relevant(self, query: str) -> bool:
        q_clean = clean_text(query)
        if len(q_clean.split()) == 0:
            return False

        words = set(q_clean.split())
        has_domain_word = bool(words & DOMAIN_HINTS)
        mentions_category = any(cat in q_clean for cat in self._category_vocab)
        mentions_skill = bool(extract_skills(query))

        if words & OFF_TOPIC_WORDS and not (has_domain_word or mentions_skill):
            return False

        if has_domain_word or mentions_category or mentions_skill:
            return True

        # Fall back to corpus similarity, but only trust it if enough of the
        # query's words are actually in the corpus vocabulary (otherwise a
        # single coincidental word can produce a misleadingly high score).
        vocab = self.vectorizer.vocabulary_
        recognised = [w for w in words if w in vocab]
        if len(recognised) < MIN_RECOGNISED_WORDS:
            return False

        q_vec = self.vectorizer.transform([q_clean])
        max_sim = cosine_similarity(q_vec, self.doc_matrix).max()
        return max_sim >= RELEVANCE_THRESHOLD

    # ---- retrieval --------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        q_vec = self.vectorizer.transform([clean_text(query)])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        results = self.df.iloc[top_idx].copy()
        results["score"] = sims[top_idx]
        return results[results["score"] > 0]

    def common_skills_for(self, text: str, top_k_docs: int = 8, top_n_skills: int = 8) -> list[str]:
        """Retrieve similar resumes and return their most frequent skills —
        used to enrich feedback with 'commonly expected' skills for a role."""
        matches = self.search(text, top_k=top_k_docs)
        counts: dict[str, int] = {}
        for _, row in matches.iterrows():
            for skill in extract_skills(row["Text"]):
                counts[skill] = counts.get(skill, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [s for s, _ in ranked[:top_n_skills]]

    # ---- conversational answer ---------------------------------------
    def answer(self, query: str, top_k: int = 5) -> dict:
        if not self.is_relevant(query):
            return {
                "relevant": False,
                "message": (
                    "I can only help with resume, candidate, and hiring related "
                    "questions using the resumes in this database. Could you "
                    "please rephrase your question or ask something else?"
                ),
                "results": pd.DataFrame(),
            }

        results = self.search(query, top_k=top_k)
        if results.empty:
            return {
                "relevant": True,
                "message": (
                    "I couldn't find any resumes in the database matching that "
                    "request. Try different skills or a different role title."
                ),
                "results": results,
            }

        lines = [f"I found {len(results)} candidate(s) that match:"]
        for _, row in results.iterrows():
            lines.append(f"- **{row['Name']}** ({row['Category']}) — "
                          f"skills: {row['Skills']}")
        message = "\n".join(lines)
        return {"relevant": True, "message": message, "results": results}
