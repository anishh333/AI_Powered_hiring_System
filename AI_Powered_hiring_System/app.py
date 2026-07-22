import os
import sys
import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from sqlmodel import select
from src.core import init_observability, get_captured_spans, clear_captured_spans
from src.database import init_db, get_session, JobDescription, Candidate
from src.ingestion import process_jd, process_resume
from src.retrieval import full_matching_pipeline
from src.agents import get_recruiter_agent
from langchain_core.messages import HumanMessage

# Initialize Observability and Database
init_observability()
init_db()

# Streamlit Configuration
st.set_page_config(page_title="AI Recruitment ATS", page_icon="👔", layout="wide")

# Add some custom CSS styling for premium look
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        border-radius: 6px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("👔 AI Recruitment ATS")
st.caption("A modern recruitment Applicant Tracking System powered by local semantic search and Logfire telemetry.")

# Setup session state for chat assistant
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = get_recruiter_agent()

# ---- 1. CENTERED DATA INGESTION ZONE ----
st.write("### ⚙️ Data Ingestion Control Panel")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Job Description")
    jd_file = st.file_uploader("Upload JD (PDF)", type=["pdf"], key="jd_upload")
    if jd_file and st.button("📥 Process and Index JD", use_container_width=True):
        with st.spinner("Extracting text and parsing JD..."):
            session = next(get_session())
            jd_id = process_jd(jd_file.getvalue(), jd_file.name, session)
            if jd_id:
                st.success(f"Successfully processed JD: {jd_file.name}")
            else:
                st.error("Failed to process Job Description")
                
with col2:
    st.subheader("Upload Resume(s)")
    resume_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True, key="res_upload")
    if resume_files and st.button("📥 Process and Index Resume(s)", use_container_width=True):
        with st.spinner(f"Parsing {len(resume_files)} candidate resume(s)..."):
            session = next(get_session())
            success_count = 0
            for f in resume_files:
                res = process_resume(f.getvalue(), f.name, session)
                if res:
                    success_count += 1
            st.success(f"Successfully processed {success_count}/{len(resume_files)} resumes.")

# ---- CHECK ANALYSIS ZONE ----
if jd_file and resume_files:
    st.write("---")
    st.subheader("⚡ Quick Match Analysis")
    st.write("Both JD and Resume(s) are uploaded. Click below to process and run match analysis immediately.")
    if st.button("📊 Check Analysis", use_container_width=True, type="primary"):
        with st.spinner("Processing documents and evaluating matches..."):
            session = next(get_session())
            
            # Process JD
            jd_id = process_jd(jd_file.getvalue(), jd_file.name, session)
            
            # Process Resume(s)
            success_count = 0
            for f in resume_files:
                res = process_resume(f.getvalue(), f.name, session)
                if res:
                    success_count += 1
                    
            if jd_id and success_count > 0:
                st.success(f"Processed JD '{jd_file.name}' and {success_count} resume(s) successfully!")
                
                # Run matching pipeline
                results = full_matching_pipeline(jd_id, session)
                
                if not results:
                    st.warning("No candidates evaluated.")
                else:
                    st.write(f"### 🎯 Match Analysis Results for: **{jd_file.name}**")
                    for idx, res in enumerate(results):
                        cand = res["candidate"]
                        eval_data = res["evaluation"]
                        
                        expander_title = f"#{idx+1} {cand.filename} — Match Score: {eval_data.score}/100"
                        with st.expander(expander_title, expanded=(idx==0)):
                            st.write(f"**Core Domain:** {cand.domain} | **Experience:** {cand.years_of_experience} yrs | **Education:** {cand.education}")
                            st.write(f"**AI Match Justification:** {eval_data.justification}")
                            if eval_data.missing_skills:
                                st.write(f"**Missing Skills:** {', '.join(eval_data.missing_skills)}")
                            st.progress(eval_data.score / 100)
            else:
                st.error("Failed to process the uploaded documents.")


# ---- API KEY CONFIGURATION ----
st.write("### 🔑 API Configuration")
env_key = os.getenv("GEMINI_API_KEY", "")
api_key_input = st.text_input(
    "Enter your Gemini API Key (optional)", 
    type="password", 
    value=env_key, 
    help="Pasting an API key enables conversational LLM evaluations and recruiter chat. Otherwise, the app runs fully offline using local heuristics."
)

if api_key_input:
    os.environ["GEMINI_API_KEY"] = api_key_input
else:
    # Retain the env-set key if available, otherwise clear it
    os.environ["GEMINI_API_KEY"] = env_key if env_key else ""

st.markdown("---")

# ---- 2. APPLICATION TABS (Database Overview Removed) ----
tab_match, tab_chat, tab_logfire = st.tabs([
    "🎯 Top Candidates Match", 
    "💬 AI Assistant", 
    "🔥 Logfire Dashboard"
])

# ---- TAB 1: TOP CANDIDATES MATCH ----
with tab_match:
    st.subheader("Evaluate Candidates for a Job")
    session = next(get_session())
    jds = session.exec(select(JobDescription)).all()
    
    if not jds:
        st.info("Upload a Job Description first to begin candidate matching.")
    else:
        jd_options = {jd.filename: jd.id for jd in jds}
        selected_jd = st.selectbox("Select Job Description to Evaluate", list(jd_options.keys()))
        
        if st.button("🔍 Find and Evaluate Top Candidates", use_container_width=True):
            with st.spinner("Running semantic matching pipeline..."):
                jd_id = jd_options[selected_jd]
                results = full_matching_pipeline(jd_id, session)
                
                if not results:
                    st.warning("No candidates found in the database. Please upload resumes first.")
                else:
                    st.write(f"### Found {len(results)} evaluated candidates:")
                    for idx, res in enumerate(results):
                        cand = res["candidate"]
                        eval_data = res["evaluation"]
                        
                        expander_title = f"#{idx+1} {cand.filename} — Match Score: {eval_data.score}/100"
                        with st.expander(expander_title, expanded=(idx==0)):
                            st.write(f"**Core Domain:** {cand.domain} | **Experience:** {cand.years_of_experience} yrs | **Education:** {cand.education}")
                            st.write(f"**AI Match Justification:** {eval_data.justification}")
                            if eval_data.missing_skills:
                                st.write(f"**Missing Skills:** {', '.join(eval_data.missing_skills)}")
                            st.progress(eval_data.score / 100)

# ---- TAB 2: AI ASSISTANT ----
with tab_chat:
    st.subheader("AI Recruiter Assistant")
    st.caption("Ask questions about job descriptions, candidates, or search criteria. Examples: 'List all candidates' or 'Find python devs'")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Ask assistant..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Recruiter thinking..."):
                try:
                    inputs = {"messages": [HumanMessage(content=prompt)]}
                    response = st.session_state.agent.invoke(inputs)
                    ai_msg = response["messages"][-1].content
                    st.markdown(ai_msg)
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                except Exception as e:
                    st.error(f"Error processing recruiter query: {e}")

# ---- TAB 3: LOGFIRE DASHBOARD ----
with tab_logfire:
    st.subheader("🔥 Local Logfire Observability & Tracing")
    st.caption("This dashboard monitors real-time telemetry captured locally by Pydantic Logfire and OpenTelemetry.")
    
    # Actions row
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("🔄 Refresh Logs", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("🧹 Clear Captured Traces", use_container_width=True):
            clear_captured_spans()
            st.success("Captured logfire spans cleared.")
            st.rerun()
            
    spans = get_captured_spans()
    
    if not spans:
        st.info("No Logfire traces captured yet. Try processing JDs, uploading resumes, or querying the AI Assistant to generate tracing spans.")
    else:
        # Construct list of traces
        trace_data = []
        for s in spans:
            start_dt = datetime.datetime.fromtimestamp(s.start_time / 1_000_000_000)
            duration_ms = (s.end_time - s.start_time) / 1_000_000
            
            trace_data.append({
                "Timestamp": start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "Trace/Span Name": s.name,
                "Duration (ms)": round(duration_ms, 2),
                "Telemetry Status": s.status.status_code.name,
                "Context Metadata (Attributes)": dict(s.attributes) if s.attributes else {}
            })
            
        # Reverse to show newest spans first
        trace_data.reverse()
        
        # Display list of spans in a clean table
        df_spans = pd.DataFrame([{
            "Timestamp": t["Timestamp"],
            "Trace/Span Name": t["Trace/Span Name"],
            "Duration (ms)": t["Duration (ms)"],
            "Telemetry Status": t["Telemetry Status"]
        } for t in trace_data])
        
        st.dataframe(df_spans, use_container_width=True)
        
        # Trace Inspector
        st.write("### 🔍 Trace Detail Inspector")
        selected_span_idx = st.selectbox(
            "Select a captured span to inspect its attributes:",
            range(len(trace_data)),
            format_func=lambda idx: f"{trace_data[idx]['Timestamp']} - {trace_data[idx]['Trace/Span Name']} ({trace_data[idx]['Duration (ms)']} ms)"
        )
        
        if selected_span_idx is not None:
            span_details = trace_data[selected_span_idx]
            st.json(span_details)
