import os

from dotenv import load_dotenv

from src.resume_parser import ResumeParser
from src.chunker import TextChunker
from src.retriever import ResumeRetriever
from src.jd_analyzer import JDAnalyzer


# --------------------------------------------------
# 1. Load environment
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found.")


# --------------------------------------------------
# 2. Load resume
# --------------------------------------------------

resume_path = "tests/data/sample_resume.pdf"

parser = ResumeParser(resume_path)

resume_text = parser.extract_text()


# --------------------------------------------------
# 3. Chunk resume
# --------------------------------------------------

chunker = TextChunker(
    chunk_size=500,
    overlap=100
)

chunks = chunker.split_text(resume_text)


print("\nResume length:", len(resume_text))
print("Number of chunks:", len(chunks))


# --------------------------------------------------
# 4. Create retriever
# --------------------------------------------------

retriever = ResumeRetriever(chunks)


# --------------------------------------------------
# 5. Create JD analyzer
# --------------------------------------------------

jd_analyzer = JDAnalyzer(
    retriever=retriever,
    api_key=api_key
)


# --------------------------------------------------
# 6. Job description
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
# 7. Analyze
# --------------------------------------------------

analysis = jd_analyzer.analyze(
    job_description=job_description,
    k=5
)


# --------------------------------------------------
# 8. Display result
# --------------------------------------------------

print("\n" + "=" * 60)
print("JD ANALYSIS")
print("=" * 60)


print("\nMATCHING STRENGTHS:")

for item in analysis.matching_strengths:
    print("-", item)


print("\nREQUIREMENTS NOT EVIDENCED:")

for item in analysis.requirements_not_evidenced:
    print("-", item)


print("\nEVIDENCE:")

for item in analysis.evidence:
    print("-", item)


print("\nSUMMARY:")
print(analysis.summary)
