from src.jd_parser import JDParser

with open(
    "data/job_descriptions/software_engineer.txt",
    "r",
    encoding="utf-8"
) as file:

    jd = file.read()

parser = JDParser(jd)

print(parser.extract_all())