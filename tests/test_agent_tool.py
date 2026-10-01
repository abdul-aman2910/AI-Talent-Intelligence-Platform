import os
import json

from dotenv import load_dotenv
from google import genai

from src.resume_parser import ResumeParser
from src.chunker import TextChunker
from src.retriever import ResumeRetriever
from src.tools import ResumeTools


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found")


# --------------------------------------------------
# 2. Prepare resume
# --------------------------------------------------

resume_path = "tests/data/sample_resume.pdf"

parser = ResumeParser(resume_path)

resume_text = parser.extract_text()


# --------------------------------------------------
# 3. Chunk resume
# --------------------------------------------------

chunker = TextChunker(
    chunk_size=500,
    overlap=100
)

chunks = chunker.split_text(resume_text)


# --------------------------------------------------
# 4. Create semantic retriever
# --------------------------------------------------

retriever = ResumeRetriever(chunks)


# --------------------------------------------------
# 5. Create resume tools
# --------------------------------------------------

resume_tools = ResumeTools(retriever)


# --------------------------------------------------
# 6. Create Gemini client
# --------------------------------------------------

client = genai.Client(
    api_key=api_key
)


# --------------------------------------------------
# 7. Define tools available to the AI agent
# --------------------------------------------------

tools = [
    {
        "type": "function",
        "name": "search_resume",
        "description": (
            "Searches the candidate's resume semantically "
            "and returns relevant resume evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The information to search for "
                        "in the candidate's resume."
                    )
                },
                "k": {
                    "type": "integer",
                    "description": (
                        "Number of relevant resume chunks "
                        "to retrieve."
                    )
                }
            },
            "required": ["query"]
        }
    }
]


# --------------------------------------------------
# 8. Recruiter question
# --------------------------------------------------

question = """
Does the candidate have experience with AWS Glue?

Use the search_resume tool to find evidence
from the candidate's resume before answering.
"""


# --------------------------------------------------
# 9. Ask Gemini
# --------------------------------------------------

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=question,
    tools=tools
)


print("\n" + "=" * 60)
print("INITIAL INTERACTION")
print("=" * 60)

print("Status:", interaction.status)


# --------------------------------------------------
# 10. Find function calls
# --------------------------------------------------

tool_results = []

for step in interaction.steps:

    if step.type != "function_call":
        continue

    print("\n" + "=" * 60)
    print("TOOL CALL")
    print("=" * 60)

    print("Tool name:", step.name)
    print("Arguments:", step.arguments)

    # ----------------------------------------------
    # Read arguments selected by Gemini
    # ----------------------------------------------

    arguments = step.arguments

    query = arguments.get(
        "query",
        question
    )

    k = arguments.get(
        "k",
        3
    )

    # ----------------------------------------------
    # Execute our actual Python tool
    # ----------------------------------------------

    print("\nExecuting search_resume()...")

    tool_result = resume_tools.search_resume(
        query=query,
        k=k
    )

    print("\n" + "=" * 60)
    print("TOOL RESULT")
    print("=" * 60)

    print(
        json.dumps(
            tool_result,
            indent=2
        )
    )

    # ----------------------------------------------
    # Prepare result for Gemini
    # ----------------------------------------------

    tool_results.append(
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
    )


# --------------------------------------------------
# 11. Send tool result back to Gemini
# --------------------------------------------------

if tool_results:

    print("\n" + "=" * 60)
    print("SENDING TOOL RESULT BACK TO GEMINI")
    print("=" * 60)

    final_interaction = client.interactions.create(
        model="gemini-3.6-flash",
        previous_interaction_id=interaction.id,
        input=tool_results
    )


    # --------------------------------------------------
    # 12. Final answer
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(
        final_interaction.output_text
    )

    print("\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)

    print(
        final_interaction.status
    )

else:

    print("\nGemini did not request any tool.")
