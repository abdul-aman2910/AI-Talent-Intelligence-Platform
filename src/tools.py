class ResumeTools:
    """
    AI tools that operate on the semantic resume retrieval system.
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def search_resume(self, query, k=3):
        """
        Search the resume semantically and return
        the most relevant resume evidence.
        """

        results = self.retriever.search(
            query,
            k=k
        )

        evidence = []

        for result in results:
            evidence.append({
                "rank": result["rank"],
                "content": result["chunk"],
                "distance": result["distance"]
            })

        return {
            "query": query,
            "results": evidence
        }