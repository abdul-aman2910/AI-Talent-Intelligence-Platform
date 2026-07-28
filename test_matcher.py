from src.resume_parser import ResumeParser
from src.information_extractor import InformationExtractor
from src.jd_parser import JDParser
from src.matcher import ResumeMatcher

# Resume
resume_parser = ResumeParser("data/resumes/sample_resume.pdf")
resume_text = resume_parser.extract_text()

resume_data = InformationExtractor(
    resume_text
).extract_all()

# Job Description
with open(
    "data/job_descriptions/software_engineer.txt",
    "r",
    encoding="utf-8"
) as f:

    jd_text = f.read()

jd_data = JDParser(jd_text).extract_all()

# Matching
matcher = ResumeMatcher(
    resume_data,
    jd_data,
    resume_text,
    jd_text
)

print("=" * 60)

print("Resume Information")
print(resume_data)

print("=" * 60)

print("Job Description")
print(jd_data)

print("=" * 60)

print("Skill Matching")
print(matcher.skill_match())

print("=" * 60)

print("Similarity")
print(matcher.text_similarity())

print("=" * 60)

print("Overall Score")
print(matcher.overall_score())