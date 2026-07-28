from src.resume_parser import ResumeParser
from src.information_extractor import InformationExtractor

parser = ResumeParser("data/resumes/sample_resume.pdf")

text = parser.extract_text()

extractor = InformationExtractor(text)

print("Name :", extractor.extract_name())
print("Email :", extractor.extract_email())
print("Phone :", extractor.extract_phone())
print("Skills :", extractor.extract_skills())
print("Education :", extractor.extract_education())
print("Experience :", extractor.extract_experience())