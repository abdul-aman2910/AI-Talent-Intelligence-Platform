from pydantic import BaseModel


class RAGResponse(BaseModel):
    answer: str
    evidence: list[str]
    sufficient_evidence: bool


class CandidateAnalysis(BaseModel):
    strengths: list[str]
    potential_areas_to_investigate: list[str]
    evidence: list[str]


class JDAnalysis(BaseModel):
    matching_strengths: list[str]
    requirements_not_evidenced: list[str]
    evidence: list[str]
    summary: str


class InterviewQuestion(BaseModel):
    question: str
    topic: str
    reason: str


class InterviewQuestionSet(BaseModel):
    questions: list[InterviewQuestion]