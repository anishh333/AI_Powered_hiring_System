"""
Text cleaning and skill-extraction utilities.

No external / Hugging Face models are used here on purpose: skill extraction is
done with a curated vocabulary + regex matching, which is fast, fully local,
and needs no model download.
"""
import re

# --- Curated skill vocabulary -------------------------------------------------
# Grouped only for readability; stored as one flat set at import time.
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

# Longer phrases must be checked before their substrings (e.g. "ruby on rails"
# before "ruby"), so sort by length descending for matching.
_SKILLS_BY_LENGTH = sorted(SKILL_VOCAB, key=len, reverse=True)


def clean_text(text: str) -> str:
    """Lowercase and normalise whitespace/punctuation for comparison."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\+#/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text: str) -> set:
    """Return the set of known skills found in the given text."""
    cleaned = f" {clean_text(text)} "
    found = set()
    for skill in _SKILLS_BY_LENGTH:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"s?(?![a-z0-9])"
        if re.search(pattern, cleaned):
            found.add(skill)
    return found


def looks_like_resume_or_jd(text: str, min_words: int = 30) -> bool:
    """Cheap sanity check that an uploaded PDF actually contains usable text."""
    if not text:
        return False
    return len(text.split()) >= min_words
