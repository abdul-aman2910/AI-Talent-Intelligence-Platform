import json

from google import genai

from src.schemas import RAGResponse


class ResumeRAG:
    """
    Retrieval-Augmented Generation over a candidate's resume.
    """

    def __init__(self, retriever, api_key):
        self.retriever = retriever

        if not api_key:
            raise ValueError("Google API Key not found.")

        self.client = genai.Client(api_key=api_key)

    def answer(self, question, k=3):

        # 1. Retrieve relevant resume chunks
        results = self.retriever.search(
            question,
            k=k
        )

        # 2. Build resume context
        context_parts = []

        for result in results:
            context_parts.append(
                f"[Resume Evidence {result['rank']}]\n"
                f"{result['chunk']}"
            )

        context = "\n\n".join(context_parts)

        # 3. Create grounded prompt
        prompt = f"""
You are an AI recruiter assistant.

Answer the question using ONLY the resume evidence provided below.

Rules:
1. Do not invent information.
2. Do not assume information that is not present.
3. If the evidence does not support the answer,
   set sufficient_evidence to false.
4. When sufficient_evidence is false, clearly state
   that there is insufficient evidence.
5. Evidence must contain specific information from
   the provided resume.
6. If the answer is supported, provide the relevant
   resume evidence in the evidence field.

Resume Evidence:
{context}

Question:
{question}
"""

        # 4. Ask Gemini for structured JSON
        interaction = self.client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            response_format=RAGResponse.model_json_schema()
        )

        # 5. Parse Gemini's JSON response
        response_data = json.loads(
            interaction.output_text
        )

        # 6. Validate response using Pydantic
        response = RAGResponse.model_validate(
            response_data
        )

        # 7. Return structured response + retrieved sources
        return {
            "response": response,
            "sources": results
        }