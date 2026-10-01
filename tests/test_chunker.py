from src.chunker import TextChunker


def test_text_chunking():
    text = """
    Built ETL pipelines using AWS Glue and Amazon S3.
    Developed a flight delay analysis platform using PySpark and Apache Spark.
    Created dashboards using Power BI and Tableau.
    Completed B.Tech in Computer Science.
    """

    chunker = TextChunker(
        chunk_size=100,
        overlap=20
    )

    chunks = chunker.split_text(text)

    # The text should be split into multiple chunks.
    assert len(chunks) > 1

    # Every chunk should contain text.
    assert all(chunk.strip() for chunk in chunks)

    # Important information should still exist after chunking.
    combined_text = " ".join(chunks)

    assert "AWS Glue" in combined_text
    assert "PySpark" in combined_text
    assert "Power BI" in combined_text