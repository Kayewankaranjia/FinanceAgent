from pydantic import BaseModel, Field
from typing import List
import json
import os
os.chdir(r"c:\Work\Tomoro\FinanceAgent")
from src.CypherGenerator.utils.cypher_utils import CypherUtils
from src.CypherGenerator.models.data_model import DocumentData, QA, ReasoningStep

class CypherGenerator:
    def __init__(self, data: DocumentData):
        self.data = data

    def generate_sentences(self) -> str:
        statements = []
        doc_id = self.data.id

        for sentence_type, sentences in [('pre', self.data.pre_text), ('post', self.data.post_text)]:
            for idx, sentence in enumerate(sentences):
                sentence_id = f"{doc_id}_{sentence_type}_{idx}"
                statements.append(CypherUtils.create_node("Sentence", {"id": sentence_id, "text": sentence, "type": sentence_type}))
                rel_type = "HAS_PRE_TEXT" if sentence_type == "pre" else "HAS_POST_TEXT"
                statements.append(CypherUtils.create_relationship("Document", doc_id, "Sentence", sentence_id, rel_type))

        return "\n".join(statements)

    def generate_table_structure(self) -> str:
        statements = []
        doc_id = self.data.id
        table_id = f"{doc_id}_table"

        statements.append(CypherUtils.create_node("Table", {"id": table_id}))
        statements.append(CypherUtils.create_relationship("Document", doc_id, "Table", table_id, "HAS_TABLE"))

        for row_idx, row in enumerate(self.data.table):
            row_id = f"{table_id}_row_{row_idx}"
            row_props = {f"col{i}": col for i, col in enumerate(row)}
            row_props["id"] = row_id

            statements.append(CypherUtils.create_node("TableRow", row_props))
            statements.append(CypherUtils.create_relationship("Table", table_id, "TableRow", row_id, "HAS_ROW"))

            for col_idx, value in enumerate(row):
                cell_id = f"{row_id}_col_{col_idx}"
                cell_props = {"id": cell_id, "value": value, "row_id": row_id, "col": col_idx, "row_index": row_idx}
                statements.append(CypherUtils.create_node("Cell", cell_props))
                statements.append(CypherUtils.create_relationship("TableRow", row_id, "Cell", cell_id, "HAS_CELL"))

                if col_idx > 0:
                    prev_cell_id = f"{row_id}_col_{col_idx - 1}"
                    statements.append(CypherUtils.create_relationship("Cell", prev_cell_id, "Cell", cell_id, "NEXT_IN_ROW"))

                if row_idx > 0:
                    above_cell_id = f"{table_id}_row_{row_idx - 1}_col_{col_idx}"
                    statements.append(CypherUtils.create_relationship("Cell", above_cell_id, "Cell", cell_id, "NEXT_IN_COLUMN"))

        return "\n".join(statements)

    def generate_qa_nodes(self) -> str:
        statements = []
        doc_id = self.data.id
        qa = self.data.qa
        qa_id = f"{doc_id}_qa"

        qa_props = {"id": qa_id, "question": qa.question, "answer": qa.answer}
        statements.append(CypherUtils.create_node("QA", qa_props))
        statements.append(CypherUtils.create_relationship("Document", doc_id, "QA", qa_id, "HAS_QA"))

        for row_idx in qa.ann_table_rows:
            row_id = f"{doc_id}_table_row_{row_idx}"
            statements.append(CypherUtils.create_relationship("QA", qa_id, "TableRow", row_id, "USES_ROW"))

        for idx, step in enumerate(qa.steps):
            step_id = f"{qa_id}_step_{idx}"
            step_props = {"id": step_id, "step": idx, "operation": step.op, "arg1": step.arg1, "arg2": step.arg2, "result": step.res}
            statements.append(CypherUtils.create_node("ReasoningStep", step_props))
            statements.append(CypherUtils.create_relationship("QA", qa_id, "ReasoningStep", step_id, "HAS_STEP"))

        return "\n".join(statements)

    def generate(self) -> str:
        statements = [
            CypherUtils.create_node("Document", {"id": self.data.id, "filename": self.data.filename}),
            self.generate_sentences(),
            self.generate_table_structure(),
            self.generate_qa_nodes()
        ]
        return "\n".join(statements)
