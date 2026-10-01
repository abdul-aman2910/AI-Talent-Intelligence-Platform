from src.resume_parser import ResumeParser


def test_resume_parser():
    resume_path = "tests/data/sample_resume.pdf"

    parser = ResumeParser(resume_path)
    text = parser.extract_text()

    assert text
    assert "Python" in text
    assert "PySpark" in text
    assert "AWS Glue" in text