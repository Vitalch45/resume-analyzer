import streamlit as st
from utils.parser import extract_text_from_pdf, analyze_resume, get_job_roles

st.title("🧠 AI Resume Analyzer")

resume_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste the Job Description here:")
job_role = st.selectbox("Select Job Role", options=get_job_roles())

if st.button("Analyze"):
    if resume_file and job_desc:
        resume_text = extract_text_from_pdf(resume_file)
        result = analyze_resume(resume_text, job_desc, job_role)

        st.subheader("📊 Resume Analysis Result")
        st.write("**Match Score:**", f"{result['score']} / 100")
        st.write("**Missing Keywords:**", result['missing_keywords'])
        st.write("**Tips:**", result['tips'])

        if result.get("gpt_suggestions"):
            st.write("**AI Suggestions:**")
            st.markdown(result['gpt_suggestions'])
    else:
        st.warning("Please upload both resume and job description!")