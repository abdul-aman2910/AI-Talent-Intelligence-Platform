import os
import json

from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# 1. Load API key
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found.")


# --------------------------------------------------
# 2. Create Gemini client
# --------------------------------------------------

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# 3. Define the actual Python tool
# --------------------------------------------------

def get_candidate_skills():
    """
    Returns the candidate's technical skills.
    """

    return {
        "skills": [
            "Python",
            "PySpark",
            "Apache Spark",
            "AWS Glue",
            "AWS S3",
            "AWS EMR",
            "Amazon Athena",
            "Boto3",
            "Terraform",
            "GitHub Actions",
            "Scikit-learn",
            "FastAPI",
            "spaCy",
            "Tableau",
            "Power BI",
            "SQL"
        ]
    }


# --------------------------------------------------
# 4. Define the tool for Gemini
# --------------------------------------------------

tools = [
    {
        "type": "function",
        "name": "get_candidate_skills",
        "description": (
            "Returns the technical skills listed "
            "in the candidate's resume."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


# --------------------------------------------------
# 5. Ask Gemini to use the tool
# --------------------------------------------------

question = """
What technical skills does the candidate have?

Use the get_candidate_skills tool to retrieve
the candidate's skills.
"""

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=question,
    tools=tools
)


# --------------------------------------------------
# 6. Display initial interaction
# --------------------------------------------------

print("\n" + "=" * 70)
print("INITIAL INTERACTION")
print("=" * 70)

print("Status:", interaction.status)


# --------------------------------------------------
# 7. Handle Gemini's tool request
# --------------------------------------------------

if interaction.status == "requires_action":

    for step in interaction.steps:

        if step.type == "function_call":

            print("\n" + "=" * 70)
            print("TOOL CALL")
            print("=" * 70)

            print("Tool name:", step.name)
            print("Arguments:", step.arguments)

            # --------------------------------------
            # Execute the requested Python function
            # --------------------------------------

            if step.name == "get_candidate_skills":

                tool_result = get_candidate_skills()

            else:

                raise ValueError(
                    f"Unknown tool requested: {step.name}"
                )

            print("\nTool result:")
            print(tool_result)

            # --------------------------------------
            # Send tool result back to Gemini
            # --------------------------------------

            interaction = client.interactions.create(
                model="gemini-3.6-flash",
                previous_interaction_id=interaction.id,
                input=[
                    {
                        "type": "function_result",
                        "name": step.name,
                        "call_id": step.id,
                        "result": [
                            {
                                "type": "text",
                                "text": json.dumps(tool_result)
                            }
                        ]
                    }
                ],
                tools=tools
            )


# --------------------------------------------------
# 8. Display final answer
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)

print(interaction.output_text)


# --------------------------------------------------
# 9. Display final status
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL STATUS")
print("=" * 70)

print(interaction.status)
