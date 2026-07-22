"""
Conversational Recruiter Agent using LangChain interfaces and Gemini REST calls.
"""
import os
import requests
from langchain_core.messages import AIMessage, HumanMessage
from sqlmodel import Session, select
import logfire

from .database import engine, Candidate, JobDescription, DB_PATH
from .rag_search import ResumeSearchEngine


class RecruiterAgent:
    """Agent class exposing an .invoke() method matching LangChain interface."""
    
    def invoke(self, inputs: dict) -> dict:
        messages = inputs.get("messages", [])
        prompt = messages[-1].content if messages else ""

        with logfire.span("Agent Invocation: {prompt}", prompt=prompt[:60]):
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                logfire.warning("GEMINI_API_KEY is missing in agent invocation.")
                return {
                    "messages": messages + [
                        AIMessage(content="Error: GEMINI_API_KEY is not configured in the environment. Please add it to your .env file.")
                    ]
                }

            # 1. RAG Search check: If candidate query, delegate to ResumeSearchEngine
            search_engine = ResumeSearchEngine(DB_PATH)
            if search_engine.is_relevant(prompt):
                logfire.info("Routing query to ResumeSearchEngine (RAG).")
                history_list = []
                for msg in messages:
                    role = "user" if isinstance(msg, HumanMessage) else "assistant"
                    history_list.append((role, msg.content))
                
                res = search_engine.answer(prompt, chat_history=history_list, api_key=api_key)
                return {
                    "messages": messages + [AIMessage(content=res["message"])]
                }

            # 2. Database query: Gather active database objects for context
            logfire.info("Routing query to General DB Recruiter Agent.")
            db_summary = self._get_db_summary()
            system_prompt = (
                "You are an AI Recruiter Assistant inside a recruitment ATS application.\n"
                "You help the user check the database status, look up job descriptions, and view candidates.\n"
                f"Here is a summary of the current SQLite database state:\n{db_summary}\n\n"
                "Answer the user's questions professionally, concisely, and helpfully."
            )

            # Build Gemini chat request body
            contents = []
            for msg in messages[:-1]:
                role = "user" if isinstance(msg, HumanMessage) else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
            contents.append({
                "role": "user",
                "parts": [{"text": system_prompt + f"\n\nUser Question: {prompt}"}]
            })

            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={api_key}"
                resp = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": contents,
                        "generationConfig": {
                            "temperature": 0.4,
                            "maxOutputTokens": 4000
                        }
                    },
                    timeout=30
                )
                resp.raise_for_status()
                ai_msg_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                return {
                    "messages": messages + [AIMessage(content=ai_msg_text)]
                }
            except Exception as e:
                logfire.error("Error querying Gemini API in agents: {error}", error=str(e))
                return {
                    "messages": messages + [
                        AIMessage(content=f"Error processing agent query: {e}")
                    ]
                }

    def _get_db_summary(self) -> str:
        """Fetch descriptive summary of the database content."""
        try:
            with Session(engine) as session:
                jds = session.exec(select(JobDescription)).all()
                cands = session.exec(select(Candidate)).all()
                
                jd_list = [f"- JD ID {jd.id}: {jd.filename} (Domain: {jd.domain})" for jd in jds]
                cand_list = [f"- Candidate ID {c.id}: {c.filename} (Domain: {c.domain}, Exp: {c.years_of_experience} yrs)" for c in cands]
                
                summary = f"Total JDs: {len(jds)} | Total Candidates: {len(cands)}\n\n"
                if jd_list:
                    summary += "Job Descriptions in system:\n" + "\n".join(jd_list) + "\n\n"
                if cand_list:
                    summary += "Candidates in system:\n" + "\n".join(cand_list) + "\n\n"
                return summary
        except Exception:
            return "No candidates or job descriptions in the database."


def get_recruiter_agent():
    return RecruiterAgent()
