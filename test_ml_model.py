from src.resume_parser import ResumeParser
from src.information_extractor import InformationExtractor
from src.jd_parser import JDParser
from src.matcher import ResumeMatcher
from src.ml_model import HiringPredictionModel

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

scores = matcher.overall_score()

# ML Model
model = HiringPredictionModel()

model.train()

result = model.predict(

    scores["skill_score"],

    scores["similarity_score"],

    scores["experience_score"],

    scores["education_score"]

)

print("=" * 60)
print(scores)
print("=" * 60)
print(result)