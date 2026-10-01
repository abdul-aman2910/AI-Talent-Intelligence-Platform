import json

from google import genai

from src.schemas import InterviewQuestionSet


class InterviewQuestionGenerator:
    """
    Generates candidate-specific interview questions
    using resume evidence and a job description.
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
        Retrieve resume evidence relevant to the
        job description.
        """

        return self.retriever.search(
            job_description,
            k=k
        )

    # ==================================================
    # CONTEXT
    # ==================================================

    def build_context(
        self,
        job_description,
        k=5
    ):
        """
        Build the context used by Gemini.
        """

        results = self.retrieve_relevant_evidence(
            job_description,
            k=k
        )

        resume_evidence = []

        for result in results:

            resume_evidence.append(
                {
                    "rank": result["rank"],
                    "content": result["chunk"],
                    "distance": result["distance"]
                }
            )

        return {
            "job_description": job_description,
            "resume_evidence": resume_evidence
        }

    # ==================================================
    # GENERATION
    # ==================================================

    def generate_questions(
        self,
        job_description,
        k=5
    ):
        """
        Generate candidate-specific interview questions
        using Gemini and retrieved resume evidence.
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
You are an AI recruiter generating technical interview
questions for a candidate.

Generate interview questions using BOTH:

1. The job description
2. The candidate's resume evidence

IMPORTANT RULES:

1. Questions should be specific to this candidate
   and this job description.

2. Prefer questions about technologies, projects,
   and concepts that are explicitly evidenced in
   the resume.

3. Questions should allow the interviewer to verify
   the candidate's actual technical understanding.

4. You may include questions about important JD
   requirements that are NOT evidenced in the resume,
   but phrase them as validation questions.

5. Do not invent candidate experience.

6. Do not claim that the candidate has a skill merely
   because it appears in the job description.

7. For every question, provide:
   - question
   - topic
   - reason

8. Generate 5 interview questions.

9. Keep questions practical and interview-oriented.

JOB DESCRIPTION:

{job_description}


RESUME EVIDENCE:

{evidence_text}
"""

        interaction = self.client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            response_format=InterviewQuestionSet.model_json_schema()
        )

        data = json.loads(
            interaction.output_text
        )

        return InterviewQuestionSet.model_validate(
            data
        )

    # ==================================================
    # VALIDATION
    # ==================================================

    def validate_questions(self, questions):
        """
        Validate generated interview questions using Pydantic.
        """

        return InterviewQuestionSet.model_validate(
            questions
        )