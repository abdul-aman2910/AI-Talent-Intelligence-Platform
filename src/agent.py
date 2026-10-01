import json

from google import genai

from src.schemas import CandidateAnalysis


class RecruiterAgent:
    """
    AI Recruiter Agent.

    Acts as the orchestration layer for the recruitment
    intelligence system.

    Responsibilities:
    - Resume semantic search
    - Candidate analysis
    - JD-aware reasoning
    - Evidence grounding
    """

    def __init__(
        self,
        resume_tools,
        api_key,
        jd_analyzer=None,
        interview_generator=None
    ):

        self.resume_tools = resume_tools
        self.jd_analyzer = jd_analyzer
        self.interview_generator = interview_generator

        if not api_key:
            raise ValueError("Google API Key not found.")

        self.client = genai.Client(
            api_key=api_key
        )

        # --------------------------------------------------
        # Tool definition
        # --------------------------------------------------

        self.tools = [
            {
                "type": "function",
                "name": "search_resume",
                "description": (
                    "Search the candidate's resume semantically "
                    "and return relevant resume evidence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A semantic search query describing "
                                "the information to find in the resume."
                            )
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

        # --------------------------------------------------
        # Agent instructions
        # --------------------------------------------------

        self.instructions = """
You are an AI recruiter assistant.

Your job is to analyze candidates using information
from their resumes.

IMPORTANT RULES:

1. Use the search_resume tool when candidate-specific
   information is required.

2. Base candidate-specific claims ONLY on information
   retrieved from the resume.

3. Never invent skills, technologies, projects,
   education, experience, or achievements.

4. If something is not mentioned or not evidenced in
   the resume, do not claim that the candidate lacks
   the skill.

5. Instead, describe it as:
   "not evidenced in the resume"
   or
   "requires validation".

6. Distinguish between:
   - demonstrated strengths
   - information that requires further investigation

7. Avoid repeated searches for the same information.

8. For broad candidate analysis, prefer a small number
   of broad semantic searches rather than many narrow searches.

9. Once sufficient evidence has been retrieved, stop
   using tools and provide the final answer.

10. Keep answers concise and recruiter-friendly.
"""

    # ==================================================
    # TOOL EXECUTION
    # ==================================================

    def _execute_tool(self, step):

        if step.name != "search_resume":
            return None

        query = step.arguments["query"]

        print("\nTOOL CALL:")
        print(step.name)

        print("Query:")
        print(query)

        print("\nExecuting search_resume...")

        result = self.resume_tools.search_resume(
            query=query,
            k=5
        )

        print("\nTOOL RESULT:")
        print(json.dumps(result, indent=2))

        return {
            "type": "function_result",
            "name": step.name,
            "call_id": step.id,
            "result": [
                {
                    "type": "text",
                    "text": json.dumps(result)
                }
            ]
        }

    # ==================================================
    # GENERIC AGENT
    # ==================================================

    def ask(self, question):

        interaction = self.client.interactions.create(
            model="gemini-3.6-flash",
            input=question,
            system_instruction=self.instructions,
            tools=self.tools
        )

        while interaction.status == "requires_action":

            tool_results = []

            for step in interaction.steps:

                if step.type != "function_call":
                    continue

                tool_result = self._execute_tool(step)

                if tool_result:
                    tool_results.append(
                        tool_result
                    )

            if not tool_results:
                break

            interaction = self.client.interactions.create(
                model="gemini-3.6-flash",
                previous_interaction_id=interaction.id,
                input=tool_results
            )

        return interaction.output_text

    # ==================================================
    # CANDIDATE ANALYSIS
    # ==================================================

    def analyze_candidate(self):

        question = """
Analyze this candidate's technical strengths based ONLY
on information supported by the candidate's resume.

Use the search_resume tool to retrieve relevant resume
evidence.

Perform the analysis using broad categories such as:

- programming and data engineering
- cloud and big data
- machine learning and AI
- backend development
- DevOps / infrastructure
- projects and technical implementation

Also identify technologies or capabilities that would be
useful to investigate further if they are not evidenced
in the resume.

IMPORTANT:

- Do not invent skills or experience.
- Do not assume that absence of a technology means the
  candidate is weak at it.
- Use "not evidenced in the resume" or
  "requires validation" where appropriate.
- Prefer broad semantic searches.
- Avoid repeatedly searching for the same information.
- Once sufficient evidence has been retrieved, stop
  searching and produce the final structured analysis.
"""

        sources = []

        interaction = self.client.interactions.create(
            model="gemini-3.6-flash",
            input=question,
            system_instruction=self.instructions,
            tools=self.tools
        )

        while interaction.status == "requires_action":

            tool_results = []

            for step in interaction.steps:

                if step.type != "function_call":
                    continue

                tool_result = self._execute_tool(step)

                if tool_result:

                    result_data = json.loads(
                        tool_result["result"][0]["text"]
                    )

                    sources.extend(
                        result_data["results"]
                    )

                    tool_results.append(
                        tool_result
                    )

            if not tool_results:
                break

            interaction = self.client.interactions.create(
                model="gemini-3.6-flash",
                previous_interaction_id=interaction.id,
                input=tool_results,
                response_format=CandidateAnalysis.model_json_schema()
            )

        if not interaction.output_text:
            raise RuntimeError(
                "Agent finished without producing a final answer."
            )

        data = json.loads(
            interaction.output_text
        )

        analysis = CandidateAnalysis.model_validate(
            data
        )

        return {
            "analysis": analysis,
            "sources": sources
        }

    # ==================================================
    # COMPLETE RECRUITER WORKFLOW
    # ==================================================

    def analyze_candidate_for_role(
        self,
        job_description
    ):
        """
        Run the complete recruiter workflow.

        Produces:
        - Candidate analysis
        - JD analysis
        - Interview questions
        """

        if self.jd_analyzer is None:
            raise ValueError(
                "JDAnalyzer is required for role analysis."
            )

        if self.interview_generator is None:
            raise ValueError(
                "InterviewQuestionGenerator is required "
                "for role analysis."
            )

        print("\n" + "=" * 60)
        print("STEP 1: CANDIDATE ANALYSIS")
        print("=" * 60)

        candidate_result = self.analyze_candidate()

        print("\nCandidate analysis completed.")

        # --------------------------------------------------
        # JD Analysis
        # --------------------------------------------------

        print("\n" + "=" * 60)
        print("STEP 2: JD ANALYSIS")
        print("=" * 60)

        jd_analysis = self.jd_analyzer.analyze(
            job_description,
            k=5
        )

        print("JD analysis completed.")

        # --------------------------------------------------
        # Interview Questions
        # --------------------------------------------------

        print("\n" + "=" * 60)
        print("STEP 3: INTERVIEW QUESTIONS")
        print("=" * 60)

        interview_questions = (
            self.interview_generator.generate_questions(
                job_description,
                k=5
            )
        )

        print("Interview question generation completed.")

        # --------------------------------------------------
        # Final result
        # --------------------------------------------------

        return {
            "candidate_analysis": candidate_result,
            "jd_analysis": jd_analysis,
            "interview_questions": interview_questions
        }