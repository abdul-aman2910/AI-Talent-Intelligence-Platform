from src.resume_parser import ResumeParser
from src.chunker import TextChunker
from src.embeddings import EmbeddingModel


# -----------------------------
# 1. Extract resume text
# -----------------------------

resume_path = "tests/data/sample_resume.pdf"

parser = ResumeParser(resume_path)
resume_text = parser.extract_text()


# -----------------------------
# 2. Split resume into chunks
# -----------------------------

chunker = TextChunker(
    chunk_size=500,
    overlap=100
)

chunks = chunker.split_text(resume_text)

print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Generate embeddings
# -----------------------------

embedding_model = EmbeddingModel()

embeddings = embedding_model.generate_embeddings(chunks)


# -----------------------------
# 4. Display information
# -----------------------------

print("Embedding shape:", embeddings.shape)

print("\nFirst chunk:")
print(chunks[0])

print("\nFirst embedding:")
print(embeddings[0])
