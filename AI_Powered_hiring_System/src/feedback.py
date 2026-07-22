"""
Generates explainable feedback for a resume against a job description.

Two modes:
  - Heuristic (default, no API key, always available): rule-based summary
    built from the match score and skill gap, enriched with a short list of
    "commonly expected skills" retrieved from similar resumes already in the
    database (a small retrieval-augmented step).
  - LLM-assisted (optional): if the user supplies an API key, the same
    retrieved context is passed to an LLM to write a more natural summary.
    This is optional and the app works fully without it.
"""
import json
import requests


def _score_band(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.5:
        return "moderate"
    if score >= 0.3:
        return "weak"
    return "poor"


def generate_heuristic_feedback(match_result: dict, retrieved_common_skills: list[str]) -> dict:
    score = match_result["final_score"]
    band = _score_band(score)
    matching = match_result["matching_skills"]
    missing = match_result["missing_skills"]

    # Skills seen often in similar roles in the database, that the candidate
    # doesn't have and weren't even explicitly asked for in the JD text.
    suggested = [s for s in retrieved_common_skills
                 if s not in matching and s not in missing][:5]

    if band == "strong":
        opener = "This is a strong match for the role."
    elif band == "moderate":
        opener = "This is a reasonable match with some notable gaps."
    elif band == "weak":
        opener = "This candidate only partially matches the role's requirements."
    else:
        opener = "This resume does not align well with the job description."

    parts = [opener]
    if matching:
        parts.append(f"The candidate demonstrates {len(matching)} of the skills "
                      f"explicitly mentioned in the job description: "
                      f"{', '.join(matching[:8])}.")
    if missing:
        parts.append(f"Missing or unconfirmed skills from the JD: "
                      f"{', '.join(missing[:8])}.")
    if suggested:
        parts.append("Candidates who matched similar roles in our database "
                      f"also commonly list: {', '.join(suggested)}.")

    summary = " ".join(parts)

    return {
        "summary": summary,
        "matching": matching,
        "missing": missing,
        "suggested": suggested,
        "band": band,
    }


def generate_llm_feedback(resume_text: str, jd_text: str, match_result: dict,
                           retrieved_common_skills: list[str], api_key: str) -> dict:
    """Optional: use an LLM (Gemini) for a more natural-language summary, grounded in the same retrieved context used by
    the heuristic path. Falls back to heuristic feedback on any error."""
    heuristic = generate_heuristic_feedback(match_result, retrieved_common_skills)

    prompt = (
        "You are a recruiting assistant. Based ONLY on the facts below, write a "
        "3-4 sentence, encouraging-but-honest evaluation of the candidate for "
        "this role. Do not invent skills that aren't listed.\n\n"
        f"Match score: {match_result['final_score']*100:.0f}%\n"
        f"Matching skills: {', '.join(match_result['matching_skills']) or 'none'}\n"
        f"Missing skills: {', '.join(match_result['missing_skills']) or 'none'}\n"
        f"Commonly expected skills for similar roles: "
        f"{', '.join(retrieved_common_skills) or 'none'}\n"
    )

    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 1500
                }
            },
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        heuristic["summary"] = text
    except Exception:
        # Silently fall back to the heuristic summary; skills lists stay valid.
        pass

    return heuristic
