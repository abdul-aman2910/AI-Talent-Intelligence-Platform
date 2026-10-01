import json

from src.schemas import CandidateAnalysis


def test_candidate_analysis_schema():
    """
    Tests that Gemini-style structured output can be
    validated correctly using the Pydantic schema.
    """

    sample_response = {
        "strengths": [
            "Python",
            "PySpark",
            "AWS Glue"
        ],
        "potential_areas_to_investigate": [
            "Kafka",
            "Snowflake",
            "Airflow"
        ],
        "evidence": [
            "Experience with Python and PySpark",
            "Built a platform using AWS Glue and Apache Spark"
        ]
    }

    analysis = CandidateAnalysis.model_validate(
        sample_response
    )

    assert isinstance(analysis, CandidateAnalysis)
    assert "Python" in analysis.strengths
    assert "PySpark" in analysis.strengths
    assert "Kafka" in analysis.potential_areas_to_investigate
    assert len(analysis.evidence) > 0