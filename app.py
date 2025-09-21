# app.py
import streamlit as st
from resume_matcher import optimize_resume
import os

st.set_page_config(page_title="AI Resume Matcher", page_icon="📄", layout="centered")

st.title("📄 AI Resume & Job Description Matcher")
st.markdown("Paste the job description and your resume (or copy/paste summary + experience). App will extract keywords, compute an ATS-like match score, and generate an optimized summary & skills section.")

job_description = st.text_area("Job Description", height=220)
resume_text = st.text_area("Your Resume (plain text)", height=220)

if st.button("Optimize Resume"):
    if not job_description.strip() or not resume_text.strip():
        st.warning("Please paste both Job Description and Resume text.")
    else:
        with st.spinner("Optimizing — calling LLM..."):
            result = optimize_resume(job_description, resume_text)

        st.subheader("🔎 ATS-like Match Score")
        score = result["score"]
        st.metric(label="Match Score", value=f"{score}%")
        st.progress(score / 100)

        st.subheader("🏷️ Extracted Keywords (from JD)")
        st.write(result["keywords"])

        st.columns(2)[0].subheader("✅ Matched")
        st.write(result["matched"])
        st.columns(2)[1].subheader("❌ Missing / Suggested")
        st.write(result["missing"])

        st.subheader("✨ Optimized Summary & Skills (Markdown)")
        st.markdown(result["optimized_markdown"])

        st.download_button(
            label="Download Optimized Resume (.docx)",
            data=result["optimized_docx_bytes"],
            file_name="optimized_resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

st.sidebar.header("Setup")
st.sidebar.write("Make sure environment variable `OPENAI_API_KEY` is set before running.")
st.sidebar.write(f"Using model: `{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}` (change via env OPENAI_MODEL)")
