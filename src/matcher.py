from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeMatcher:

    def __init__(self, resume_data, jd_data, resume_text, jd_text):

        self.resume_data = resume_data
        self.jd_data = jd_data
        self.resume_text = resume_text
        self.jd_text = jd_text

    # ------------------------------------
    # Skill Match
    # ------------------------------------

    def skill_match(self):

        resume_skills = set(
            skill.lower() for skill in self.resume_data["skills"]
        )

        jd_skills = set(
            skill.lower() for skill in self.jd_data["skills"]
        )

        matched = resume_skills.intersection(jd_skills)

        missing = jd_skills - resume_skills

        if len(jd_skills) == 0:
            percentage = 0.0
        else:
            percentage = float(
                round(
                    (len(matched) / len(jd_skills)) * 100,
                    2
                )
            )

        return {

            "matched_skills": sorted(matched),

            "missing_skills": sorted(missing),

            "skill_match_percentage": percentage

        }

    # ------------------------------------
    # Resume Similarity
    # ------------------------------------

    def text_similarity(self):

        vectorizer = TfidfVectorizer(stop_words="english")

        vectors = vectorizer.fit_transform(
            [self.resume_text, self.jd_text]
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return float(round(similarity * 100, 2))

    # ------------------------------------
    # Experience Score
    # ------------------------------------

    def experience_score(self):

        resume_exp = self.resume_data["experience"]

        jd_exp = self.jd_data["experience"]

        if jd_exp == 0:
            return 100.0

        if resume_exp >= jd_exp:
            return 100.0

        return float(
            round(
                (resume_exp / jd_exp) * 100,
                2
            )
        )

    # ------------------------------------
    # Education Score
    # ------------------------------------

    def education_score(self):

        resume = set(
            x.lower() for x in self.resume_data["education"]
        )

        jd = set(
            x.lower() for x in self.jd_data["education"]
        )

        if len(jd) == 0:
            return 100.0

        if resume.intersection(jd):
            return 100.0

        return 0.0

    # ------------------------------------
    # Final Score
    # ------------------------------------

    def overall_score(self):

        skill = self.skill_match()["skill_match_percentage"]

        similarity = self.text_similarity()

        experience = self.experience_score()

        education = self.education_score()

        final = float(
            round(
                (skill * 0.45) +
                (similarity * 0.30) +
                (experience * 0.15) +
                (education * 0.10),
                2
            )
        )

        return {

            "overall_score": final,

            "skill_score": skill,

            "similarity_score": similarity,

            "experience_score": experience,

            "education_score": education

        }