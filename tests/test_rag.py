import os

from dotenv import load_dotenv

from src.resume_parser import ResumeParser
from src.chunker import TextChunker
from src.retriever import ResumeRetriever
from src.rag import ResumeRAG


# --------------------------------------------------
# 1. Load environment variables
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

print("=" * 70)
print("RESUME")
print("=" * 70)

print("Resume text length:", len(resume_text))


# --------------------------------------------------
# 3. Split resume into chunks
# --------------------------------------------------

chunker = TextChunker(
    chunk_size=500,
    overlap=100
)

chunks = chunker.split_text(resume_text)

print("Number of chunks:", len(chunks))


# --------------------------------------------------
# 4. Create semantic retriever
# --------------------------------------------------

retriever = ResumeRetriever(chunks)

print("Retriever created successfully.")


# --------------------------------------------------
# 5. Create RAG system
# --------------------------------------------------

rag = ResumeRAG(
    retriever=retriever,
    api_key=api_key
)

print("RAG system created successfully.")


# --------------------------------------------------
# 6. Ask question
# --------------------------------------------------

question = "Does the candidate have experience with AWS Glue?"

result = rag.answer(
    question,
    k=3
)


# --------------------------------------------------
# 7. Display RAG response
# --------------------------------------------------

print("\n" + "=" * 70)
print("QUESTION")
print("=" * 70)

print(question)


print("\n" + "=" * 70)
print("RAG ANSWER")
print("=" * 70)

response = result["response"]


print("\nAnswer:")
print(response.answer)


print("\nSufficient evidence:")
print(response.sufficient_evidence)


print("\nEvidence:")

if response.evidence:
    for evidence in response.evidence:
        print("-", evidence)
else:
    print("No supporting evidence found.")


# --------------------------------------------------
# 8. Display retrieved sources
# --------------------------------------------------

print("\n" + "=" * 70)
print("RETRIEVED SOURCES")
print("=" * 70)

for source in result["sources"]:

    print(f"\nSource {source['rank']}")

    print("Distance:", source["distance"])

    print(source["chunk"])


print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)
