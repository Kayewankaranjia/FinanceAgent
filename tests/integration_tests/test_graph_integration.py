import pytest
from agent import graph
from unittest.mock import patch, MagicMock
import pandas as pd

@pytest.mark.integration
def test_full_graph_flow(monkeypatch):
    # Mock Neo4j embedding and query
    monkeypatch.setattr(graph, "openai_embeddings", MagicMock(embed_query=lambda x: [0.1, 0.2]))
    mock_result = MagicMock()
    mock_record = MagicMock()
    mock_record.data.return_value = {"filename": "testfile.csv"}
    mock_result.records = [mock_record]
    monkeypatch.setattr(graph.driver, "execute_query", lambda *a, **kw: mock_result)

    # Mock LLM responses for each step
    class MockLLM:
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            # Return different responses based on the prompt
            last_msg = messages[-1] if messages else None
            if hasattr(last_msg, "content") and "component" in last_msg.content.lower():
                return MagicMock(content="{'OperationToBePerformed': 'percentage change', 'RequiredTimeFrames': [2007, 2008], 'Metric': ['capital expenditures']}")
            if hasattr(last_msg, "content") and "formula" in last_msg.content.lower():
                return MagicMock(content="(capex_2008 - capex_2007) / capex_2007 * 100")
            if hasattr(last_msg, "content") and "calculator" in last_msg.content.lower():
                return MagicMock(content="-27.66")
            if hasattr(last_msg, "content") and "final" in last_msg.content.lower():
                return MagicMock(content="The percentage change in capital expenditures from 2007 to 2008 is approximately -27.66%.")
            return MagicMock(content="ok")
    monkeypatch.setattr(graph, "llm", MockLLM())
    monkeypatch.setattr(graph, "llm_with_tool", MockLLM())

    # Mock data loading
    df = pd.DataFrame({
        "filename": ["testfile.csv"],
        "capital expenditures": [100, 72.34],
        "year": [2007, 2008]
    })
    monkeypatch.setattr(pd, "read_json", lambda *a, **k: df)

    # Prepare initial state (simulate a user question)
    initial_state = {
        "messages": [
            MagicMock(
                content="what is the percentage change in capital expenditures from 2007 to 2008?",
                artifact=None,
                tool_calls=[]
            )
        ]
    }

    # Run the full graph
    result = graph.graph.invoke(initial_state)

    # The final message should contain the expected answer
    final_message = result["messages"][-1]
    assert "The percentage change in capital expenditures from 2007 to 2008 is approximately -27.66%" in str(final_message.content)