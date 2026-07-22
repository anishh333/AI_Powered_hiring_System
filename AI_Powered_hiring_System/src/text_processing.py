"""
Text cleaning and heuristic extraction utilities.

No external / Hugging Face models or LLMs are used here on purpose: extraction is
done with curated vocabularies + regex matching, which is fast, fully local,
and needs zero API cost.
"""
import re
from typing import Set

# --- Curated skill vocabulary -------------------------------------------------
SKILL_GROUPS = {
    "languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl",
        "sql", "html", "css", "bash", "shell scripting", "dart",
    ],
    "frameworks": [
        "django", "flask", "fastapi", "spring", "spring boot", "react", "angular",
        "vue", "next.js", "node.js", "express", "laravel", "ruby on rails",
        ".net", "asp.net", "hibernate", "jquery", "bootstrap", "tensorflow",
        "pytorch", "keras", "scikit-learn", "pandas", "numpy", "streamlit",
    ],
    "databases": [
        "mysql", "postgresql", "mongodb", "oracle", "sql server", "sqlite",
        "redis", "cassandra", "dynamodb", "elasticsearch", "mariadb", "neo4j",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform",
        "jenkins", "ansible", "ci/cd", "git", "github", "gitlab", "linux",
        "cloudformation", "helm", "prometheus", "grafana", "nginx", "vagrant",
    ],
    "data_ml": [
        "machine learning", "deep learning", "nlp", "computer vision",
        "data analysis", "data visualization", "big data", "spark", "hadoop",
        "etl", "power bi", "tableau", "airflow", "kafka", "data engineering",
        "statistics", "a/b testing",
    ],
    "practices": [
        "agile", "scrum", "kanban", "microservices", "rest api", "graphql",
        "object oriented programming", "unit testing", "tdd", "devops",
        "system design", "api development", "mvc",
    ],
    "soft_skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "project management", "time management", "critical thinking",
        "collaboration", "mentoring", "stakeholder management",
    ],
}

SKILL_VOCAB = sorted({s.lower() for group in SKILL_GROUPS.values() for s in group})
_SKILLS_BY_LENGTH = sorted(SKILL_VOCAB, key=len, reverse=True)


def clean_text(text: str) -> str:
    """Lowercase and normalise whitespace/punctuation for comparison."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\+#/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text: str) -> Set[str]:
    """Return the set of known skills found in the given text."""
    cleaned = f" {clean_text(text)} "
    found = set()
    for skill in _SKILLS_BY_LENGTH:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"s?(?![a-z0-9])"
        if re.search(pattern, cleaned):
            found.add(skill)
    return found


def extract_years_of_experience(text: str) -> int:
    """Heuristic extraction of years of experience."""
    cleaned = text.lower()
    # Match patterns like "5+ years of experience", "10 yrs", "2 years"
    pattern = r'(\d{1,2})\+?\s*(?:years?|yrs?)(?:\s*of)?\s*(?:experience|exp)'
    matches = re.findall(pattern, cleaned)
    if matches:
        years = [int(m) for m in matches if int(m) < 60]
        if years:
            return max(years)
    return 0


def extract_education_level(text: str) -> str:
    """Heuristic extraction of highest education level."""
    cleaned = text.lower()
    if re.search(r'\b(phd|doctorate|d\.phil)\b', cleaned):
        return "PhD"
    elif re.search(r'\b(master|m\.s\.|ms|m\.a\.|ma|mba|msc)\b', cleaned):
        return "Masters"
    elif re.search(r'\b(bachelor|b\.s\.|bs|b\.a\.|ba|bsc|btech)\b', cleaned):
        return "Bachelors"
    elif re.search(r'\b(associate)\b', cleaned):
        return "Associates"
    return "Unknown"


def infer_domain(skills_found: Set[str]) -> str:
    """Infer domain by finding which skill group has the most matches."""
    group_counts = {group: 0 for group in SKILL_GROUPS.keys()}
    for skill in skills_found:
        for group, group_skills in SKILL_GROUPS.items():
            if skill in group_skills:
                group_counts[group] += 1
                break
    
    if not skills_found:
        return "Unknown"
        
    best_group = max(group_counts, key=group_counts.get)
    if group_counts[best_group] == 0:
        return "Unknown"
        
    return best_group.replace("_", " ").title()


def extract_summary(text: str) -> str:
    """Extract a simple summary (e.g., first 300 characters)."""
    idx = text.lower().find("summary")
    if idx != -1 and idx < 1000:
        return text[idx:idx+400].strip()
    return text[:400].strip()


def looks_like_resume_or_jd(text: str, min_words: int = 30) -> bool:
    """Cheap sanity check that an uploaded PDF actually contains usable text."""
    if not text:
        return False
    return len(text.split()) >= min_words
