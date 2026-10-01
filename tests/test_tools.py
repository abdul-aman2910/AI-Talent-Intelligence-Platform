from src.resume_parser import ResumeParser
from src.chunker import TextChunker
from src.retriever import ResumeRetriever
from src.tools import ResumeTools


def test_search_resume_tool():
    # 1. Load resume
    resume_path = "tests/data/sample_resume.pdf"

    parser = ResumeParser(resume_path)
    resume_text = parser.extract_text()

    assert resume_text

    # 2. Chunk resume
    chunker = TextChunker(
        chunk_size=500,
        overlap=100
    )

    chunks = chunker.split_text(resume_text)

    assert len(chunks) > 0

    # 3. Create retriever
    retriever = ResumeRetriever(chunks)

    # 4. Create resume tools
    tools = ResumeTools(retriever)

    # 5. Search using the tool
    query = "Does the candidate have experience with AWS Glue?"

    k = min(3, len(chunks))

    result = tools.search_resume(
        query,
        k=k
    )

    # 6. Validate tool response structure
    assert result["query"] == query
    assert "results" in result
    assert len(result["results"]) == k

    # 7. Validate each retrieved result
    for rank, item in enumerate(result["results"], start=1):
        assert item["rank"] == rank
        assert item["content"]
        assert isinstance(item["distance"], float)

    # 8. Validate relevant evidence
    retrieved_text = " ".join(
        item["content"]
        for item in result["results"]
    )

    assert "AWS Glue" in retrieved_text