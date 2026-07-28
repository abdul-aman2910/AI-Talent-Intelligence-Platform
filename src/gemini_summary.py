import os
from google import genai
from dotenv import load_dotenv

load_dotenv()


class GeminiSummary:

    def __init__(self, resume_data, jd_data, match_result, ml_result):

        self.resume = resume_data
        self.jd = jd_data
        self.match = match_result
        self.ml = ml_result

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("Google API Key not found.")

        self.client = genai.Client(api_key=api_key)

    def generate_summary(self):

        prompt = f"""
You are an experienced technical recruiter.

Analyze the candidate based on the information below.

Candidate Name:
{self.resume.get("name")}

Candidate Skills:
{", ".join(self.resume.get("skills", []))}

Education:
{", ".join(self.resume.get("education", []))}

Experience:
{self.resume.get("experience")} years

Job Required Skills:
{", ".join(self.jd.get("skills", []))}

Matched Skills:
{", ".join(self.match.get("matched_skills", []))}

Missing Skills:
{", ".join(self.match.get("missing_skills", []))}

Skill Match:
{self.match.get("skill_match_percentage")}%

Prediction:
{self.ml.get("prediction")}

Selection Probability:
{self.ml.get("selection_probability")}%

Write a professional recruiter summary.

Include:
- Candidate strengths
- Weaknesses
- Missing skills
- Final hiring recommendation

Keep it under 150 words.
"""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text