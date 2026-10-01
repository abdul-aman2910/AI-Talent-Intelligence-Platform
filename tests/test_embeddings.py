from src.embeddings import EmbeddingModel


def test_embedding_generation():
    texts = [
        "Built ETL pipelines using AWS Glue and Amazon S3.",
        "Developed a flight delay analysis platform using PySpark.",
        "Completed B.Tech in Computer Science."
    ]

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.generate_embeddings(texts)

    # We provided 3 texts, so we should get 3 embeddings.
    assert embeddings.shape[0] == 3

    # all-MiniLM-L6-v2 produces 384-dimensional embeddings.
    assert embeddings.shape[1] == 384

    # The embeddings should contain numerical values.
    assert embeddings is not None