from pydantic import BaseModel, Field
from typing import List


# === Data Models ===

class ReasoningStep(BaseModel):
    op: str
    arg1: str
    arg2: str
    res: str


class QA(BaseModel):
    question: str
    answer: str
    ann_table_rows: List[int] = Field(default_factory=list)
    steps: List[ReasoningStep] = Field(default_factory=list)


class DocumentData(BaseModel):
    id: str
    filename: str
    pre_text: List[str]
    post_text: List[str]
    table: List[List[str]]
    qa: QA = None
