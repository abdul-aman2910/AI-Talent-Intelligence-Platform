import os

import pytest
from dotenv import load_dotenv
from google import genai


load_dotenv()


@pytest.mark.integration
def test_gemini_connection():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not configured")

    client = genai.Client(api_key=api_key)

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input="Say hello and explain in one sentence what you are."
    )

    assert interaction.output_text