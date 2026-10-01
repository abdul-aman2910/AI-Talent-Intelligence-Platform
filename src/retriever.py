from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class ResumeRetriever:
    """
    Handles semantic search over resume chunks.
    """

    def __init__(self, chunks):
        self.chunks = chunks

        # Load embedding model
        self.embedding_model = EmbeddingModel()

        # Generate embeddings for resume chunks
        embeddings = self.embedding_model.generate_embeddings(chunks)

        # Create FAISS vector store
        dimension = embeddings.shape[1]
        self.vector_store = VectorStore(dimension)

        # Store embeddings in FAISS
        self.vector_store.add_embeddings(embeddings)

    def search(self, query, k=3):
        """
        Search the resume for chunks relevant to the query.
        """

        # Convert query into embedding
        query_embedding = self.embedding_model.generate_embeddings(
            [query]
        )

        # Search FAISS
        distances, indices = self.vector_store.search(
            query_embedding,
            k
        )

        results = []

        for rank, index in enumerate(indices[0]):

            results.append({
                "rank": rank + 1,
                "chunk": self.chunks[index],
                "distance": float(distances[0][rank])
            })

        return results