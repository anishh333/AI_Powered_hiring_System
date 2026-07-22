"""
Retrieval and candidate matching pipeline.
"""
import os
from types import SimpleNamespace
from sqlmodel import Session, select
import logfire

from .database import JobDescription, Candidate, DB_PATH
from .matching import SemanticMatcher, evaluate_match
from .feedback import generate_heuristic_feedback, generate_llm_feedback
from .rag_search import ResumeSearchEngine


def full_matching_pipeline(jd_id: int, session: Session) -> list[dict]:
    """Retrieves all candidates, runs the matching evaluation, and sorts them."""
    with logfire.span("Executing retrieval pipeline for JD ID: {jd_id}", jd_id=jd_id) as span:
        # 1. Fetch the Job Description
        jd = session.get(JobDescription, jd_id)
        if not jd:
            logfire.warning("Job Description with ID {jd_id} not found", jd_id=jd_id)
            return []

        span.set_attribute("jd_filename", jd.filename)
        span.set_attribute("jd_domain", jd.domain)

        # 2. Fetch all candidates from Candidate table
        candidates = session.exec(select(Candidate)).all()
        if not candidates:
            logfire.info("No candidates found in database to evaluate.")
            return []

        span.set_attribute("candidates_count", len(candidates))

        # 3. Load background corpus for TF-IDF training
        logfire.info("Loading reference resume corpus from database...")
        search_engine = ResumeSearchEngine(DB_PATH)
        corpus = search_engine.df["Text"].fillna("").tolist()
        
        # Add new candidate texts to background corpus
        for cand in candidates:
            if cand.raw_text:
                corpus.append(cand.raw_text)

        # 4. Initialize matcher with background corpus
        logfire.info("Fitting TF-IDF vectorizer on background corpus...")
        matcher = SemanticMatcher(corpus)

        results = []
        api_key = os.getenv("GEMINI_API_KEY", "")

        # 5. Evaluate each candidate
        for cand in candidates:
            with logfire.span("Evaluating candidate: {filename}", filename=cand.filename) as cand_span:
                match_result = evaluate_match(matcher, cand.raw_text, jd.raw_text)
                
                # Set evaluation statistics on the candidate span
                cand_span.set_attribute("candidate_id", cand.id)
                cand_span.set_attribute("match_score", int(match_result["final_score"] * 100))
                cand_span.set_attribute("semantic_score", match_result["semantic_score"])
                cand_span.set_attribute("skill_score", match_result["skill_score"])
                cand_span.set_attribute("matching_skills_count", len(match_result["matching_skills"]))
                cand_span.set_attribute("missing_skills_count", len(match_result["missing_skills"]))

                common_skills = search_engine.common_skills_for(jd.raw_text)

                if api_key:
                    feedback = generate_llm_feedback(
                        cand.raw_text, jd.raw_text, match_result, common_skills, api_key
                    )
                else:
                    feedback = generate_heuristic_feedback(match_result, common_skills)

                # Return SimpleNamespace to support attribute access in app.py
                eval_data = SimpleNamespace(
                    score=int(match_result["final_score"] * 100),
                    justification=feedback["summary"],
                    missing_skills=match_result["missing_skills"]
                )

                results.append({
                    "candidate": cand,
                    "evaluation": eval_data
                })

        # 6. Sort by evaluation score descending
        results.sort(key=lambda x: x["evaluation"].score, reverse=True)
        logfire.info("Successfully evaluated {count} candidates", count=len(results))
        return results
