import io
import fitz  # PyMuPDF
import docx
import spacy
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# --- NLP Model Loading ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- FastAPI App Initialization ---
app = FastAPI(title="SkillSync API")

# --- CORS Middleware Configuration ---
# This allows your frontend (hosted on Netlify) to communicate with your backend (hosted on Render).
origins = ["*"]  # Allows all origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Predefined Skill Lists (Expanded) ---
# (You can continue to add more skills and aliases here)
HARD_SKILLS = {
    'python', 'java', 'c++', 'c#', 'javascript', 'js', 'typescript', 'ts', 'html', 'css', 'sql', 'nosql',
    'react', 'angular', 'vue', 'node.js', 'nodejs', 'django', 'flask', 'fastapi', 'spring', 'springboot',
    'aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes', 'git', 'github', 'gitlab', 'jenkins',
    'jira', 'agile', 'scrum', 'machine learning', 'ml', 'deep learning', 'tensorflow', 'pytorch',
    'pandas', 'numpy', 'scikit-learn', 'data analysis', 'data science', 'api', 'rest', 'graphql',
    'mongodb', 'postgresql', 'mysql', 'redis', 'kafka', 'linux', 'bash', 'shell scripting', 'devops'
}
SOFT_SKILLS = {
    'communication', 'teamwork', 'leadership', 'problem-solving', 'problem solving', 'critical thinking',
    'creativity', 'adaptability', 'time management', 'collaboration', 'work ethic', 'interpersonal skills',
    'conflict resolution', 'negotiation', 'mentorship', 'presentation skills'
}
ACTION_VERBS = {
    'developed', 'led', 'managed', 'created', 'implemented', 'designed', 'architected', 'optimized',
    'automated', 'streamlined', 'improved', 'increased', 'decreased', 'reduced', 'achieved'
}

# --- Helper Functions ---
def extract_text(file_stream, filename: str) -> str:
    """Extracts text from PDF or DOCX files."""
    if filename.endswith(".pdf"):
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_stream.read()))
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .pdf or .docx file.")

def find_skills(doc, skill_list):
    """Finds skills from a predefined list in a spaCy Doc."""
    found_skills = set()
    for token in doc:
        if token.lower_ in skill_list:
            found_skills.add(token.lower_)
    # Check for multi-word skills (e.g., 'data analysis')
    for skill in skill_list:
        if ' ' in skill and skill in doc.text.lower():
            found_skills.add(skill)
    return list(found_skills)

def generate_resume_tips(resume_doc):
    """Generates actionable tips based on resume content."""
    tips = []
    # Tip 1: Check for action verbs
    if not any(token.lemma_.lower() in ACTION_VERBS for token in resume_doc):
        tips.append("Strengthen your resume by starting bullet points with strong action verbs (e.g., 'developed', 'managed', 'implemented').")
    
    # Tip 2: Check for quantifiable achievements
    if not any(ent.label_ == "CARDINAL" or ent.label_ == "PERCENT" for ent in resume_doc.ents):
        tips.append("Include quantifiable achievements to show impact. Use numbers and percentages to highlight your accomplishments (e.g., 'Increased efficiency by 20%').")

    # Tip 3: Always add a proofreading tip
    tips.append("Always proofread your resume for any spelling or grammar errors before submitting. A clean, error-free resume looks professional.")
    
    return tips


# --- API Endpoint ---
@app.post("/analyze/")
async def analyze_resume(
    resume: UploadFile = File(...), 
    job_description: str = Form(...)
):
    """
    Analyzes a resume against a job description, providing a match score,
    skill gap analysis, and actionable tips.
    """
    try:
        resume_text = extract_text(resume.file, resume.filename)
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=500, detail="Error processing the uploaded file.")

    # Process texts with spaCy
    resume_doc = nlp(resume_text)
    jd_doc = nlp(job_description)

    # 1. Calculate Match Percentage (Semantic Similarity)
    similarity_score = resume_doc.similarity(jd_doc)
    match_percentage = round(similarity_score * 100, 1)

    # 2. Skill Gap Analysis
    jd_hard_skills = find_skills(jd_doc, HARD_SKILLS)
    jd_soft_skills = find_skills(jd_doc, SOFT_SKILLS)
    all_jd_skills = set(jd_hard_skills + jd_soft_skills)

    resume_hard_skills = find_skills(resume_doc, HARD_SKILLS)
    resume_soft_skills = find_skills(resume_doc, SOFT_SKILLS)
    all_resume_skills = set(resume_hard_skills + resume_soft_skills)
    
    matched_skills = list(all_jd_skills.intersection(all_resume_skills))
    missing_skills = list(all_jd_skills.difference(all_resume_skills))

    # 3. Generate Actionable Tips
    resume_tips = generate_resume_tips(resume_doc)
    
    # --- Prepare and Return Response ---
    return {
        "filename": resume.filename,
        "match_percentage": match_percentage,
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
        "resume_tips": resume_tips,
        "detail": f"The resume is a {match_percentage}% match with the job description."
    }

