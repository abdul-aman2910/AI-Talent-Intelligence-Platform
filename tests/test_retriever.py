from src.resume_parser import ResumeParser
from src.chunker import TextChunker
from src.retriever import ResumeRetriever


def test_resume_retrieval():
    # 1. Extract resume text
    resume_path = "tests/data/sample_resume.pdf"

    parser = ResumeParser(resume_path)
    resume_text = parser.extract_text()

    assert resume_text

    # 2. Create resume chunks
    chunker = TextChunker(
        chunk_size=500,
        overlap=100
    )

    chunks = chunker.split_text(resume_text)

    assert len(chunks) > 0

    # 3. Create retriever
    retriever = ResumeRetriever(chunks)

    # 4. Search the resume
    query = "Does the candidate have experience with AWS Glue?"

    k = min(2, len(chunks))

    results = retriever.search(
        query,
        k=k
    )

    # 5. Validate retrieval results
    assert len(results) == k

    for rank, result in enumerate(results, start=1):
        assert result["rank"] == rank
        assert result["chunk"]
        assert isinstance(result["distance"], float)

    # 6. Check that relevant evidence was retrieved
    retrieved_text = " ".join(
        result["chunk"] for result in results
    )

    assert "AWS Glue" in retrieved_text