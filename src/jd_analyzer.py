import json

from google import genai

from src.schemas import JDAnalysis


class JDAnalyzer:
    """
    Analyzes a candidate's resume against a job description.

    FAISS is used for retrieving relevant resume evidence.
    Gemini is used to interpret that evidence against the JD.
    """

    def __init__(self, retriever, api_key):

        self.retriever = retriever

        if not api_key:
            raise ValueError("Google API Key not found.")

        self.client = genai.Client(
            api_key=api_key
        )

    # ==================================================
    # RETRIEVAL
    # ==================================================

    def retrieve_relevant_evidence(
        self,
        job_description,
        k=5
    ):
        """
        Retrieve resume chunks that are semantically
        relevant to the job description.
        """

        results = self.retriever.search(
            job_description,
            k=k
        )

        return results

    # ==================================================
    # BUILD CONTEXT
    # ==================================================

    def build_context(
        self,
        job_description,
        k=5
    ):
        """
        Build the context that will be given to Gemini.
        """

        results = self.retrieve_relevant_evidence(
            job_description,
            k=k
        )

        evidence = []

        for result in results:

            evidence.append(
                {
                    "rank": result["rank"],
                    "content": result["chunk"],
                    "distance": result["distance"]
                }
            )

        return {
            "job_description": job_description,
            "resume_evidence": evidence
        }

    # ==================================================
    # JD ANALYSIS
    # ==================================================

    def analyze(
        self,
        job_description,
        k=5
    ):
        """
        Analyze the candidate against the provided JD.

        Retrieval happens locally using FAISS.
        Gemini performs the final interpretation.
        """

        context = self.build_context(
            job_description,
            k=k
        )

        evidence_text = ""

        for item in context["resume_evidence"]:

            evidence_text += (
                f"\n[Resume Evidence {item['rank']}]\n"
                f"{item['content']}\n"
            )

        prompt = f"""
You are an AI recruiter assistant.

Analyze the candidate's resume against the provided
job description.

IMPORTANT:

1. Use ONLY the provided resume evidence.
2. Do not invent candidate skills or experience.
3. Identify skills and requirements that are clearly
   supported by the resume.
4. If a JD requirement is not present in the provided
   evidence, describe it as:
   "not evidenced in the resume"
   or
   "requires validation".
5. Do NOT conclude that the candidate is incapable of
   something merely because it is absent from the resume.
6. Provide concrete resume evidence for your analysis.
7. Keep the analysis concise and recruiter-friendly.

JOB DESCRIPTION:

{job_description}


RESUME EVIDENCE:

{evidence_text}
"""

        interaction = self.client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            response_format=JDAnalysis.model_json_schema()
        )

        data = json.loads(
            interaction.output_text
        )

        analysis = JDAnalysis.model_validate(
            data
        )

        return analysis