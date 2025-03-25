import fitz  # PyMuPDF
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests

nltk.download('punkt')

def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def analyze_resume(resume_text, job_desc, job_role):
    resume_text = resume_text.lower()

    role_keywords_dict = {
        "data scientist": ["machine learning", "deep learning", "python", "statistics", "data analysis", "pandas", "modeling", "numpy", "scikit-learn", "visualization", "ai"],
        "software engineer": ["java", "python", "algorithms", "data structures", "git", "api", "rest", "design patterns", "unit testing", "oop", "debugging"],
        "frontend developer": ["react", "html", "css", "javascript", "ui/ux", "redux", "responsive design", "figma", "webpack", "tailwind", "typescript"],
        "backend developer": ["django", "node.js", "api", "sql", "database", "microservices", "authentication", "flask", "orm", "postgresql", "mongodb"]
    }

    keywords_to_check = role_keywords_dict.get(job_role.lower(), [])
    resume_words = set(resume_text.split())
    missing_keywords = [kw for kw in keywords_to_check if kw not in resume_words]

    vectorizer = CountVectorizer(stop_words='english').fit([resume_text, job_desc])
    vectors = vectorizer.transform([resume_text, job_desc])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    tips = []
    if len(resume_text) < 1000:
        tips.append("Your resume is a bit short. Consider adding more detail.")
    if len(missing_keywords) > 0:
        tips.append("Add relevant skills based on the selected job role.")

    gpt_suggestions = get_gpt_suggestions(resume_text, job_desc)

    return {
        "score": round(similarity * 100),
        "missing_keywords": missing_keywords[:10],
        "tips": tips,
        "gpt_suggestions": gpt_suggestions
    }

def get_job_roles():
    return ["data scientist", "software engineer", "frontend developer", "backend developer"]

def get_gpt_suggestions(resume_text, job_desc):
    try:
        prompt = f"""
You are a helpful resume advisor. Suggest 3 ways to improve this resume to better match the job description.

Resume:
{resume_text[:2000]}

Job Description:
{job_desc[:1000]}
"""

        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()

        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Could not generate suggestions due to: {e}"