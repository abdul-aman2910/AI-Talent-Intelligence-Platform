from src.resume_parser import ResumeParser
from src.chunker import TextChunker


# Path to your resume PDF
resume_path = "tests/data/sample_resume.pdf"


# Step 1: Extract text from PDF
parser = ResumeParser(resume_path)
resume_text = parser.extract_text()

print("Resume text length:", len(resume_text))


# Step 2: Split resume text into chunks
chunker = TextChunker(
    chunk_size=500,
    overlap=100
)

chunks = chunker.split_text(resume_text)


# Step 3: Display chunks
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)
