import re


class JDParser:
    """
    Parses a Job Description and extracts
    skills, education and experience.
    """

    def __init__(self, jd_text):
        self.text = jd_text

    # -----------------------------------------
    # SKILLS
    # -----------------------------------------
    def extract_skills(self):

        jd_text = self.text.lower()

        skills = []

        try:
            with open(
                "data/skills/skills.txt",
                "r",
                encoding="utf-8"
            ) as file:

                for line in file:

                    skill = line.strip()

                    if not skill:
                        continue

                    pattern = r"\b" + re.escape(skill.lower()) + r"\b"

                    if re.search(pattern, jd_text):
                        skills.append(skill)

        except FileNotFoundError:
            print("skills.txt not found!")

        return sorted(set(skills))

    # -----------------------------------------
    # EDUCATION
    # -----------------------------------------
    def extract_education(self):

        education_keywords = [

            "Bachelor",
            "Master",
            "B.Tech",
            "BTech",
            "M.Tech",
            "MTech",
            "B.E",
            "BE",
            "M.E",
            "ME",
            "MBA",
            "MCA",
            "BCA",
            "B.Sc",
            "M.Sc",
            "Diploma",
            "PhD"

        ]

        found = []

        for keyword in education_keywords:

            if re.search(
                r"\b" + re.escape(keyword) + r"\b",
                self.text,
                re.IGNORECASE
            ):
                found.append(keyword)

        return sorted(set(found))

    # -----------------------------------------
    # EXPERIENCE
    # -----------------------------------------
    def extract_experience(self):

        patterns = [

            r'(\d+(?:\.\d+)?)\+?\s*years?',
            r'(\d+(?:\.\d+)?)\+?\s*yrs?',
            r'(\d+(?:\.\d+)?)\+?\s*year'

        ]

        experience = []

        for pattern in patterns:

            matches = re.findall(
                pattern,
                self.text,
                re.IGNORECASE
            )

            for match in matches:

                try:
                    experience.append(float(match))
                except ValueError:
                    pass

        if experience:
            return max(experience)

        return 0

    # -----------------------------------------
    # ALL
    # -----------------------------------------
    def extract_all(self):

        return {

            "skills": self.extract_skills(),

            "education": self.extract_education(),

            "experience": self.extract_experience()

        }