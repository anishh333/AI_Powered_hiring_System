import os
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from sqlmodel import select
from src.core import init_observability
from src.database import init_db, get_session, JobDescription, Candidate
from src.ingestion import process_jd, process_resume
from src.retrieval import full_matching_pipeline
from src.agents import get_recruiter_agent
from langchain_core.messages import HumanMessage

# Initialize
init_observability()
init_db()

st.set_page_config(page_title="AI Recruitment ATS", page_icon="👔", layout="wide")
st.title("👔 AI Recruitment ATS")

# Setup session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = get_recruiter_agent()

with st.sidebar:
    st.header("⚙️ Data Ingestion")
    
    st.subheader("Upload Job Description")
    jd_file = st.file_uploader("Upload JD (PDF)", type=["pdf"], key="jd_upload")
    if jd_file and st.button("Process JD"):
        with st.spinner("Processing..."):
            session = next(get_session())
            jd_id = process_jd(jd_file.getvalue(), jd_file.name, session)
            if jd_id:
                st.success(f"Processed JD: {jd_file.name}")
            else:
                st.error("Failed to process JD")
                
    st.subheader("Upload Resume(s)")
    resume_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True, key="res_upload")
    if resume_files and st.button("Process Resumes"):
        with st.spinner(f"Processing {len(resume_files)} resumes..."):
            session = next(get_session())
            success_count = 0
            for f in resume_files:
                res = process_resume(f.getvalue(), f.name, session)
                if res:
                    success_count += 1
            st.success(f"Processed {success_count}/{len(resume_files)} resumes.")

tab1, tab2, tab3 = st.tabs(["📋 Database Overview", "🎯 Top Candidates Match", "💬 AI Assistant"])

with tab1:
    session = next(get_session())
    st.subheader("Job Descriptions")
    jds = session.exec(select(JobDescription)).all()
    if not jds:
        st.write("No Job Descriptions uploaded.")
    for jd in jds:
        st.write(f"- {jd.filename} (Domain: {jd.domain})")
        
    st.subheader("Candidates")
    cands = session.exec(select(Candidate)).all()
    st.write(f"Total Candidates: {len(cands)}")
    for c in cands:
        st.write(f"- {c.filename} | Domain: {c.domain} | Exp: {c.years_of_experience} yrs")

with tab2:
    st.subheader("Evaluate Candidates for a Job")
    session = next(get_session())
    jds = session.exec(select(JobDescription)).all()
    
    if not jds:
        st.info("Upload a Job Description first.")
    else:
        jd_options = {jd.filename: jd.id for jd in jds}
        selected_jd = st.selectbox("Select Job Description", list(jd_options.keys()))
        
        if st.button("🔍 Find Top Candidates"):
            with st.spinner("Retrieving and evaluating candidates..."):
                jd_id = jd_options[selected_jd]
                results = full_matching_pipeline(jd_id, session)
                
                if not results:
                    st.warning("No candidates found.")
                else:
                    for idx, res in enumerate(results):
                        cand = res["candidate"]
                        eval_data = res["evaluation"]
                        with st.expander(f"#{idx+1} {cand.filename} - Score: {eval_data.score}/100", expanded=(idx==0)):
                            st.write(f"**Domain:** {cand.domain} | **Experience:** {cand.years_of_experience} yrs")
                            st.write(f"**Justification:** {eval_data.justification}")
                            if eval_data.missing_skills:
                                st.write(f"**Missing Skills:** {', '.join(eval_data.missing_skills)}")
                            st.progress(eval_data.score / 100)

with tab3:
    st.subheader("Recruiter Assistant")
    st.caption("E.g., 'List the job descriptions' or 'Find top candidates for <JD_ID>'")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Ask about candidates, JDs, or find matches..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    inputs = {"messages": [HumanMessage(content=prompt)]}
                    response = st.session_state.agent.invoke(inputs)
                    ai_msg = response["messages"][-1].content
                    st.markdown(ai_msg)
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                except Exception as e:
                    st.error(f"Error processing agent query: {e}")
