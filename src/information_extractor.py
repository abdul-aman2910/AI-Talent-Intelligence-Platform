import re
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


class InformationExtractor:
    """
    Extracts structured information from resume text.
    """

    def __init__(self, text):
        self.text = text

    # -------------------------------------------------
    # EMAIL
    # -------------------------------------------------
    def extract_email(self):
        """
        Extracts the first email address found in the resume.
        """
        match = re.search(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            self.text,
        )

        if match:
            return match.group()

        return None

    # -------------------------------------------------
    # PHONE
    # -------------------------------------------------
    def extract_phone(self):
        """
        Extracts phone number.
        """

        match = re.search(
            r"(?:\+91[-\s]?)?[6-9]\d{9}",
            self.text
        )

        if match:
            phone = match.group()

            phone = phone.replace("+91", "")
            phone = phone.replace("-", "")
            phone = phone.replace(" ", "")

            return phone

        return None

    # -------------------------------------------------
    # NAME
    # -------------------------------------------------
    def extract_name(self):
        """
        Extracts candidate name using spaCy.
        """

        if nlp is None:
            return None

        doc = nlp(self.text)

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()

        return None

    # -------------------------------------------------
    # SKILLS
    # -------------------------------------------------
    def extract_skills(self):
        """
        Extracts skills from skills.txt
        """

        resume_text = self.text.lower()

        found_skills = []

        try:

            with open(
                "data/skills/skills.txt",
                "r",
                encoding="utf-8"
            ) as file:

                for line in file:

                    skill = line.strip()

                    if skill == "":
                        continue

                    pattern = r"\b" + re.escape(skill.lower()) + r"\b"

                    if re.search(pattern, resume_text):
                        found_skills.append(skill)

        except FileNotFoundError:
            print("skills.txt not found!")

        return sorted(list(set(found_skills)))

    # -------------------------------------------------
    # EDUCATION
    # -------------------------------------------------
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
            "PhD",
            "Doctorate",
            "SSC",
            "HSC",
            "12th",
            "10th"

        ]

        found = []

        for keyword in education_keywords:

            if re.search(
                r"\b" + re.escape(keyword) + r"\b",
                self.text,
                re.IGNORECASE
            ):
                found.append(keyword)

        return sorted(list(set(found)))

    # -------------------------------------------------
    # EXPERIENCE
    # -------------------------------------------------
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

    # -------------------------------------------------
    # ALL DETAILS
    # -------------------------------------------------
    def extract_all(self):

        return {

            "name": self.extract_name(),

            "email": self.extract_email(),

            "phone": self.extract_phone(),

            "skills": self.extract_skills(),

            "education": self.extract_education(),

            "experience": self.extract_experience()

        }