from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os

from src.resume_parser import ResumeParser
from src.information_extractor import InformationExtractor
from src.jd_parser import JDParser
from src.matcher import ResumeMatcher
from src.ml_model import HiringPredictionModel
from src.ai_summary import RecruiterSummary
from src.gemini_summary import GeminiSummary
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AI Talent Intelligence Platform")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "data/processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={
        "request": request
    }
)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):

    # Save uploaded resume
    resume_path = os.path.join(
        UPLOAD_FOLDER,
        resume.filename
    )

    with open(resume_path, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)

    # Resume Parsing
    resume_text = ResumeParser(
        resume_path
    ).extract_text()

    resume_data = InformationExtractor(
        resume_text
    ).extract_all()

    # JD Parsing
    jd_data = JDParser(
        job_description
    ).extract_all()

    # Matching
    matcher = ResumeMatcher(
        resume_data,
        jd_data,
        resume_text,
        job_description
    )

    skill_result = matcher.skill_match()

    scores = matcher.overall_score()

    # ML Prediction
    model = HiringPredictionModel()

    model.train()

    ml_result = model.predict(
        scores["skill_score"],
        scores["similarity_score"],
        scores["experience_score"],
        scores["education_score"]
    )

    # AI Summary
    try:
        summary = GeminiSummary(
        resume_data,
        jd_data,
        skill_result,
        ml_result
        ).generate_summary()

    except Exception as e:

        print("Gemini Error:", e)

        summary = RecruiterSummary(
            resume_data,
            jd_data,
            skill_result,
            ml_result
        ).generate_summary()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "resume": resume_data,
            "match": skill_result,
            "scores": scores,
            "prediction": ml_result,
            "summary": summary
        }
    )