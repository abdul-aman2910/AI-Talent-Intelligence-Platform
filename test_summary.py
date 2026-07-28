from src.resume_parser import ResumeParser
from src.information_extractor import InformationExtractor
from src.jd_parser import JDParser
from src.matcher import ResumeMatcher
from src.ml_model import HiringPredictionModel
from src.ai_summary import RecruiterSummary

# Resume
resume_text = ResumeParser(
    "data/resumes/sample_resume.pdf"
).extract_text()

resume = InformationExtractor(
    resume_text
).extract_all()

# Job Description
with open(
    "data/job_descriptions/software_engineer.txt",
    "r",
    encoding="utf-8"
) as file:

    jd_text = file.read()

jd = JDParser(jd_text).extract_all()

# Matching
matcher = ResumeMatcher(
    resume,
    jd,
    resume_text,
    jd_text
)

match = matcher.skill_match()

scores = matcher.overall_score()

# ML
model = HiringPredictionModel()

model.train()

ml = model.predict(
    scores["skill_score"],
    scores["similarity_score"],
    scores["experience_score"],
    scores["education_score"]
)

summary = RecruiterSummary(
    resume,
    jd,
    match,
    ml
)

print(summary.generate_summary())