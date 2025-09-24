import io
import fitz  # PyMuPDF
import docx
import spacy
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# --- NLP Model Loading ---
# The model is now downloaded during the build step, so we can load it directly.
nlp = spacy.load("en_core_web_sm")

# --- FastAPI App Initialization ---
app = FastAPI(title="SkillSync API")

# --- CORS Middleware Configuration ---
origins = ["*"]  # Allows all origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Predefined Skill Lists (Expanded) ---
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
    for skill in skill_list:
        if ' ' in skill and skill in doc.text.lower():
            found_skills.add(skill)
    return list(found_skills)

def generate_resume_tips(resume_doc):
    """Generates actionable tips based on resume content."""
    tips = []
    if not any(token.lemma_.lower() in ACTION_VERBS for token in resume_doc):
        tips.append("Strengthen your resume by starting bullet points with strong action verbs (e.g., 'developed', 'managed', 'implemented').")
    if not any(ent.label_ == "CARDINAL" or ent.label_ == "PERCENT" for ent in resume_doc.ents):
        tips.append("Include quantifiable achievements to show impact. Use numbers and percentages to highlight your accomplishments (e.g., 'Increased efficiency by 20%').")
    tips.append("Always proofread your resume for any spelling or grammar errors before submitting. A clean, error-free resume looks professional.")
    return tips

# --- API Endpoint ---
@app.post("/analyze/")
async def analyze_resume(
    resume: UploadFile = File(...), 
    job_description: str = Form(...)
):
    try:
        resume_text = extract_text(resume.file, resume.filename)
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=500, detail="Error processing the uploaded file.")

    resume_doc = nlp(resume_text)
    jd_doc = nlp(job_description)

    similarity_score = resume_doc.similarity(jd_doc)
    match_percentage = round(similarity_score * 100, 1) if similarity_score > 0 else 0

    jd_skills = set(find_skills(jd_doc, HARD_SKILLS.union(SOFT_SKILLS)))
    resume_skills = set(find_skills(resume_doc, HARD_SKILLS.union(SOFT_SKILLS)))
    
    matched_skills = list(jd_skills.intersection(resume_skills))
    missing_skills = list(jd_skills.difference(resume_skills))

    resume_tips = generate_resume_tips(resume_doc)
    
    return {
        "filename": resume.filename,
        "match_percentage": match_percentage,
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
        "resume_tips": resume_tips,
        "detail": f"The resume is a {match_percentage}% match with the job description."
    }

