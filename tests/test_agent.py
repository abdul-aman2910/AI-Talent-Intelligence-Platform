import os

from dotenv import load_dotenv

from src.resume_parser import ResumeParser
from src.chunker import TextChunker
from src.retriever import ResumeRetriever
from src.tools import ResumeTools
from src.agent import RecruiterAgent
from src.jd_analyzer import JDAnalyzer
from src.interview_generator import InterviewQuestionGenerator


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found.")


# --------------------------------------------------
# Load resume
# --------------------------------------------------

resume_path = "tests/data/sample_resume.pdf"

parser = ResumeParser(resume_path)

resume_text = parser.extract_text()

print("\nResume loaded")
print("Resume length:", len(resume_text))


# --------------------------------------------------
# Chunk resume
# --------------------------------------------------

chunker = TextChunker(
    chunk_size=500,
    overlap=100
)

chunks = chunker.split_text(resume_text)

print("Number of chunks:", len(chunks))


# --------------------------------------------------
# Create retriever
# --------------------------------------------------

retriever = ResumeRetriever(chunks)


# --------------------------------------------------
# Create components
# --------------------------------------------------

resume_tools = ResumeTools(
    retriever
)

jd_analyzer = JDAnalyzer(
    retriever=retriever,
    api_key=api_key
)

interview_generator = InterviewQuestionGenerator(
    retriever=retriever,
    api_key=api_key
)


# --------------------------------------------------
# Create recruiter agent
# --------------------------------------------------

agent = RecruiterAgent(
    resume_tools=resume_tools,
    api_key=api_key,
    jd_analyzer=jd_analyzer,
    interview_generator=interview_generator
)


# --------------------------------------------------
# Job description
# --------------------------------------------------

job_description = """
We are looking for a Big Data Engineer.

Required skills:

Python, SQL, PySpark, Apache Spark,
AWS S3, AWS Glue, EMR, Kafka, Snowflake,
Airflow and Docker.

The candidate should have experience building
data pipelines, working with large datasets,
cloud data platforms and ETL systems.
"""


# --------------------------------------------------
# Run complete recruiter workflow
# --------------------------------------------------

result = agent.analyze_candidate_for_role(
    job_description
)


# ==================================================
# DISPLAY FINAL REPORT
# ==================================================

print("\n\n" + "=" * 70)
print("              AI RECRUITER REPORT")
print("=" * 70)


# --------------------------------------------------
# Candidate Analysis
# --------------------------------------------------

candidate_result = result["candidate_analysis"]

analysis = candidate_result["analysis"]

sources = candidate_result["sources"]


print("\n\nCANDIDATE STRENGTHS")
print("-" * 70)

for strength in analysis.strengths:
    print("-", strength)


print("\n\nAREAS TO INVESTIGATE")
print("-" * 70)

for area in analysis.potential_areas_to_investigate:
    print("-", area)


print("\n\nCANDIDATE EVIDENCE")
print("-" * 70)

for evidence in analysis.evidence:
    print("-", evidence)


# --------------------------------------------------
# JD Analysis
# --------------------------------------------------

jd_analysis = result["jd_analysis"]


print("\n\nJD MATCHING STRENGTHS")
print("-" * 70)

for strength in jd_analysis.matching_strengths:
    print("-", strength)


print("\n\nJD REQUIREMENTS NOT EVIDENCED")
print("-" * 70)

for requirement in jd_analysis.requirements_not_evidenced:
    print("-", requirement)


print("\n\nJD EVIDENCE")
print("-" * 70)

for evidence in jd_analysis.evidence:
    print("-", evidence)


print("\n\nJD SUMMARY")
print("-" * 70)

print(jd_analysis.summary)


# --------------------------------------------------
# Interview Questions
# --------------------------------------------------

interview_questions = result[
    "interview_questions"
]


print("\n\nINTERVIEW QUESTIONS")
print("-" * 70)

for i, question in enumerate(
    interview_questions.questions,
    start=1
):

    print(f"\n{i}. {question.question}")

    print(
        f"   Topic: {question.topic}"
    )

    print(
        f"   Reason: {question.reason}"
    )


# --------------------------------------------------
# Grounding Sources
# --------------------------------------------------

print("\n\n" + "=" * 70)
print("              GROUNDING SOURCES")
print("=" * 70)


for source in sources:

    print("\nRank:", source["rank"])

    print(
        "Distance:",
        source["distance"]
    )

    print(
        "Content:",
        source["content"]
    )
