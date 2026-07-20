import os
import sys
import glob
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(__file__))
from src.pdf_utils import extract_text_from_pdf
from src.text_processing import looks_like_resume_or_jd
from src.matching import SemanticMatcher, evaluate_match
from src.feedback import generate_heuristic_feedback, generate_llm_feedback
from src.rag_search import ResumeSearchEngine

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "resume_database.sqlite")

st.set_page_config(page_title="Resume Evaluator", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Minimal, clean theming
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .stApp { background-color: #0b1220; }
    h1, h2, h3, h4, p, span, label, li { color: #e5e7eb; }
    section[data-testid="stSidebar"] { background-color: #0f172a; }

    .card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.8em;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .pill-match { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.4); }
    .pill-missing { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.4); }
    .pill-suggest { background: rgba(129,140,248,0.15); color: #a5b4fc; border: 1px solid rgba(129,140,248,0.4); }

    .stButton>button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white; border: none; border-radius: 8px; font-weight: 600;
    }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(139,92,246,0.35); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading resume database...")
def load_engine():
    return ResumeSearchEngine(DB_PATH)


@st.cache_resource(show_spinner=False)
def load_matcher(_engine):
    return SemanticMatcher(_engine.df["Text"].tolist())


def score_gauge(score: float):
    pct = round(score * 100, 1)
    color = "#10b981" if pct >= 60 else ("#f59e0b" if pct >= 35 else "#f87171")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"color": color, "size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155"},
            "bar": {"color": color},
            "bgcolor": "#111827",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "rgba(248,113,113,0.15)"},
                {"range": [35, 60], "color": "rgba(245,158,11,0.15)"},
                {"range": [60, 100], "color": "rgba(16,185,129,0.15)"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e5e7eb"})
    return fig


def pills(items, css_class):
    if not items:
        return "<span style='color:#64748b;'>None</span>"
    return "".join(f"<span class='pill {css_class}'>{s}</span>" for s in items)


# ---------------------------------------------------------------------------
engine = load_engine()
matcher = load_matcher(engine)

st.title("📄 AI Resume Evaluator")
st.caption("Upload a job description and a resume (PDF) to get an explainable, "
           "AI-assisted match score — or ask the assistant to find candidates.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("Optional: add a Gemini API key for a more natural-language "
                 "summary and a conversational AI chat assistant. "
                 "Without it, local rule-based features are used — "
                 "the app works fully offline either way.")
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key (optional)", type="password", value=env_key)

tab_evaluate, tab_chat = st.tabs(["🎯 Evaluate a Resume", "💬 Ask the Assistant"])

# ---------------------------------------------------------------------------
# TAB 1 — Resume vs JD evaluation (PDF only)
# ---------------------------------------------------------------------------
with tab_evaluate:

    def get_pdf_files(folder):
        os.makedirs(folder, exist_ok=True)
        return [""] + [os.path.basename(f) for f in glob.glob(f"{folder}/*.pdf")]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Job Description")
        jd_files = get_pdf_files("jd")
        selected_jd_file = st.selectbox("Select saved JD", jd_files, key="sel_jd")
        jd_file = st.file_uploader("Or Upload new JD (PDF)", type=["pdf"], key="jd_pdf")
        if jd_file and st.button("💾 Save Uploaded JD", use_container_width=True):
            with open(os.path.join("jd", jd_file.name), "wb") as f:
                f.write(jd_file.getvalue())
            st.toast(f"Saved {jd_file.name}")

    with col2:
        st.subheader("Candidate Resume")
        resume_files = get_pdf_files("resume")
        selected_resume_file = st.selectbox("Select saved Resume", resume_files, key="sel_res")
        resume_file = st.file_uploader("Or Upload new Resume (PDF)", type=["pdf"], key="resume_pdf")
        if resume_file and st.button("💾 Save Uploaded Resume", use_container_width=True):
            with open(os.path.join("resume", resume_file.name), "wb") as f:
                f.write(resume_file.getvalue())
            st.toast(f"Saved {resume_file.name}")

    analyze = st.button("🔍 Analyze Resume", use_container_width=True)

    if analyze:
        jd_bytes = None
        if jd_file:
            jd_bytes = jd_file.read()
        elif selected_jd_file:
            with open(os.path.join("jd", selected_jd_file), "rb") as f:
                jd_bytes = f.read()

        resume_bytes = None
        if resume_file:
            resume_bytes = resume_file.read()
        elif selected_resume_file:
            with open(os.path.join("resume", selected_resume_file), "rb") as f:
                resume_bytes = f.read()

        if not jd_bytes or not resume_bytes:
            st.error("Please provide both a Job Description and a Resume (either uploaded or selected).")
        else:
            jd_text = extract_text_from_pdf(jd_bytes)
            resume_text = extract_text_from_pdf(resume_bytes)

            if not looks_like_resume_or_jd(jd_text) or not looks_like_resume_or_jd(resume_text):
                st.error(
                    "This request cannot be processed — one or both PDFs don't "
                    "contain enough readable text (they may be scanned images "
                    "or empty). Please upload a text-based PDF and try again."
                )
            else:
                with st.spinner("Analyzing match and generating feedback..."):
                    match_result = evaluate_match(matcher, resume_text, jd_text)
                    common_skills = engine.common_skills_for(jd_text)

                    if api_key:
                        feedback = generate_llm_feedback(
                            resume_text, jd_text, match_result, common_skills, api_key)
                    else:
                        feedback = generate_heuristic_feedback(match_result, common_skills)

                st.markdown("---")
                st.header("📊 Analysis Results")

                gcol, scol = st.columns([1, 2])
                with gcol:
                    st.plotly_chart(score_gauge(match_result["final_score"]),
                                     use_container_width=True, config={"displayModeBar": False})
                with scol:
                    st.markdown(f"<div class='card'><b>🤖 AI Summary</b><br><br>{feedback['summary']}</div>",
                                unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"<div class='card'><b>✅ Matching Skills</b><br><br>"
                                f"{pills(feedback['matching'], 'pill-match')}</div>",
                                unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='card'><b>⚠️ Missing Skills</b><br><br>"
                                f"{pills(feedback['missing'], 'pill-missing')}</div>",
                                unsafe_allow_html=True)
                with c3:
                    st.markdown(f"<div class='card'><b>💡 Also Worth Having</b><br><br>"
                                f"{pills(feedback['suggested'], 'pill-suggest')}</div>",
                                unsafe_allow_html=True)

                with st.expander("How was this generated?"):
                    st.markdown(f"""
- **Semantic similarity** ({match_result['semantic_score']*100:.1f}%): computed with a
  TF-IDF vector-space model fit on the local resume database (scikit-learn — no
  external embedding service is used).
- **Skill overlap** ({match_result['skill_score']*100:.1f}%): explicit skills detected
  in the JD vs. the resume text.
- **"Also worth having"** skills are retrieved (RAG) from similar resumes already
  in the database, so the feedback reflects real-world expectations for the role,
  not just the literal wording of the JD.
- Final score = 60% skill overlap + 40% semantic similarity.
""")

# ---------------------------------------------------------------------------
# TAB 2 — Conversational candidate search (RAG + guardrail)
# ---------------------------------------------------------------------------
with tab_chat:
    st.subheader("💬 Ask the Recruiter Assistant")
    st.caption("Ask about candidates in the database, e.g. \"Find me a Java "
               "developer who knows Python, SQL, and AWS.\" Questions unrelated "
               "to hiring or resumes will be declined.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    user_query = st.chat_input("Ask about candidates, roles, or skills...")
    if user_query:
        st.session_state.chat_history.append(("user", user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        result = engine.answer(user_query, chat_history=st.session_state.chat_history, api_key=api_key)
        with st.chat_message("assistant"):
            st.markdown(result["message"])
        st.session_state.chat_history.append(("assistant", result["message"]))
