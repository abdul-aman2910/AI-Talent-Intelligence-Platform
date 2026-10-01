import os

from dotenv import load_dotenv

from src.resume_parser import ResumeParser
from src.chunker import TextChunker
from src.retriever import ResumeRetriever
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

print("\nResume length:", len(resume_text))


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
# Create interview generator
# --------------------------------------------------

generator = InterviewQuestionGenerator(
    retriever=retriever,
    api_key=api_key
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
# Generate questions
# --------------------------------------------------

questions = generator.generate_questions(
    job_description,
    k=5
)


# --------------------------------------------------
# Display
# --------------------------------------------------

print("\n" + "=" * 60)
print("AI INTERVIEW QUESTION GENERATOR")
print("=" * 60)

for i, question in enumerate(
    questions.questions,
    start=1
):

    print(f"\n{i}. {question.question}")

    print(
        f"   Topic: {question.topic}"
    )

    print(
        f"   Reason: {question.reason}"
    )
