class RecruiterSummary:

    def __init__(self, resume_data, jd_data, match_result, ml_result):

        self.resume = resume_data
        self.jd = jd_data
        self.match = match_result
        self.ml = ml_result

    def generate_summary(self):

        name = self.resume["name"]

        matched = self.match["matched_skills"]

        missing = self.match["missing_skills"]

        score = self.match["skill_match_percentage"]

        prediction = self.ml["prediction"]

        probability = self.ml["selection_probability"]

        summary = []

        summary.append(f"Candidate: {name}")

        summary.append(f"Overall Skill Match: {score}%")

        if matched:
            summary.append(
                "Matched Skills: " +
                ", ".join(matched)
            )

        if missing:
            summary.append(
                "Missing Skills: " +
                ", ".join(missing)
            )

        summary.append(
            f"Hiring Prediction: {prediction}"
        )

        summary.append(
            f"Selection Probability: {probability}%"
        )

        if prediction == "Selected":

            summary.append(
                "The candidate satisfies most of the technical requirements and is recommended for the next round."
            )

        else:

            summary.append(
                "The candidate does not currently satisfy enough technical requirements. Learning the missing skills would significantly improve suitability for this role."
            )

        return "\n".join(summary)