import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class HiringPredictionModel:

    def __init__(self):

        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    # ------------------------------------
    # Train Model
    # ------------------------------------

    def train(self):

        df = pd.read_csv(
            "data/model/training_data.csv"
        )

        X = df[
            [
                "skill_score",
                "similarity_score",
                "experience_score",
                "education_score"
            ]
        ]

        y = df["hired"]

        self.model.fit(X, y)

    # ------------------------------------
    # Predict
    # ------------------------------------

    def predict(
        self,
        skill_score,
        similarity_score,
        experience_score,
        education_score
    ):

        sample = pd.DataFrame(
            [[
                skill_score,
                similarity_score,
                experience_score,
                education_score
            ]],
            columns=[
                "skill_score",
                "similarity_score",
                "experience_score",
                "education_score"
            ]
        )

        prediction = self.model.predict(sample)[0]

        probability = self.model.predict_proba(sample)[0][1]

        return {
            "prediction": "Selected" if prediction == 1 else "Rejected",
            "selection_probability": float(round(probability * 100, 2))
        }