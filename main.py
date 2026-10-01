from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv

import shutil
import os

from src.resume_parser import ResumeParser
from src.chunker import TextChunker
from src.retriever import ResumeRetriever
from src.tools import ResumeTools
from src.agent import RecruiterAgent
from src.jd_analyzer import JDAnalyzer
from src.interview_generator import InterviewQuestionGenerator

load_dotenv()

# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="AI Talent Intelligence Platform"
)


# ==================================================
# STATIC FILES / TEMPLATES
# ==================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(
    directory="templates"
)


# ==================================================
# FILE STORAGE
# ==================================================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==================================================
# HOME
# ==================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )


# ==================================================
# ANALYZE RESUME
# ==================================================

@app.post(
    "/analyze",
    response_class=HTMLResponse
)
async def analyze(
    request: Request,
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):

    # --------------------------------------------------
    # 1. Save uploaded resume
    # --------------------------------------------------

    resume_path = os.path.join(
        UPLOAD_FOLDER,
        resume.filename
    )

    with open(
        resume_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            resume.file,
            buffer
        )


    # --------------------------------------------------
    # 2. Extract resume text
    # --------------------------------------------------

    resume_text = ResumeParser(
        resume_path
    ).extract_text()


    # --------------------------------------------------
    # 3. Chunk resume
    # --------------------------------------------------

    chunker = TextChunker(
        chunk_size=500,
        overlap=100
    )

    chunks = chunker.split_text(
        resume_text
    )


    # --------------------------------------------------
    # 4. Create semantic retriever
    # --------------------------------------------------

    retriever = ResumeRetriever(
        chunks
    )


    # --------------------------------------------------
    # 5. Create resume tools
    # --------------------------------------------------

    resume_tools = ResumeTools(
        retriever
    )


    # --------------------------------------------------
    # 6. Get Gemini API key
    # --------------------------------------------------

    api_key = os.getenv(
        "GOOGLE_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GOOGLE_API_KEY not found."
        )


    # --------------------------------------------------
    # 7. Create JD analyzer
    # --------------------------------------------------

    jd_analyzer = JDAnalyzer(
        retriever=retriever,
        api_key=api_key
    )


    # --------------------------------------------------
    # 8. Create interview generator
    # --------------------------------------------------

    interview_generator = (
        InterviewQuestionGenerator(
            retriever=retriever,
            api_key=api_key
        )
    )


    # --------------------------------------------------
    # 9. Create recruiter agent
    # --------------------------------------------------

    agent = RecruiterAgent(
        resume_tools=resume_tools,
        api_key=api_key,
        jd_analyzer=jd_analyzer,
        interview_generator=interview_generator
    )


    # --------------------------------------------------
    # 10. Run complete AI recruiter workflow
    # --------------------------------------------------

    result = agent.analyze_candidate_for_role(
        job_description
    )


    # --------------------------------------------------
    # 11. Extract results
    # --------------------------------------------------

    candidate_result = result[
        "candidate_analysis"
    ]

    candidate_analysis = candidate_result[
        "analysis"
    ]

    sources = candidate_result[
        "sources"
    ]

    jd_analysis = result[
        "jd_analysis"
    ]

    interview_questions = result[
        "interview_questions"
    ]


    # --------------------------------------------------
    # 12. Return AI recruiter dashboard
    # --------------------------------------------------

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,

            "analysis_complete": True,

            "candidate_analysis": candidate_analysis,

            "jd_analysis": jd_analysis,

            "interview_questions": interview_questions,

            "sources": sources,

            "job_description": job_description
        }
    )