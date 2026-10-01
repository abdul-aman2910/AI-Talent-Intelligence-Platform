from src.resume_parser import ResumeParser
from src.chunker import TextChunker
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


def test_vector_store_search():
    # 1. Extract resume text
    resume_path = "tests/data/sample_resume.pdf"

    parser = ResumeParser(resume_path)
    resume_text = parser.extract_text()

    assert resume_text

    # 2. Create chunks
    chunker = TextChunker(
        chunk_size=500,
        overlap=100
    )

    chunks = chunker.split_text(resume_text)

    assert len(chunks) > 0

    # 3. Generate embeddings
    embedding_model = EmbeddingModel()

    resume_embeddings = embedding_model.generate_embeddings(chunks)

    assert resume_embeddings.shape[0] == len(chunks)
    assert resume_embeddings.shape[1] == 384

    # 4. Create FAISS vector store
    dimension = resume_embeddings.shape[1]

    vector_store = VectorStore(dimension)

    vector_store.add_embeddings(resume_embeddings)

    # Number of vectors in FAISS should equal number of chunks.
    assert vector_store.size() == len(chunks)

    # 5. Create query embedding
    query = "Does the candidate have experience with AWS Glue?"

    query_embedding = embedding_model.generate_embeddings(
        [query]
    )

    # 6. Search FAISS
    k = min(3, len(chunks))
    
    distances, indices = vector_store.search(
         query_embedding,
         k=k
    )

    # FAISS should return 3 results.
    assert distances.shape == (1, k)
    assert indices.shape == (1, k)

    # Returned indices should point to valid chunks.
    for index in indices[0]:
        assert 0 <= index < len(chunks)