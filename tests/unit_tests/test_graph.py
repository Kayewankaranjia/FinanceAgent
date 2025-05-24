import pytest
from unittest.mock import patch, MagicMock
from agent import graph
import pandas as pd

def test_embed():
    texts = ["a", "b"]
    result = graph.embed(texts)
    assert isinstance(result, list)
    assert all(isinstance(vec, list) for vec in result)

def test_user_question_tool():
    result = graph.user_question_tool("q", "op", ["tf"], ["metric"])
    assert result["question"] == "q"
    assert result["operation"] == "op"
    assert result["timeframe"] == ["tf"]
    assert result["metric"] == ["metric"]

def test_fetch_filename_from_neo4j(monkeypatch):
    monkeypatch.setattr(graph, "openai_embeddings", MagicMock(embed_query=lambda x: [0.1, 0.2]))
    mock_result = MagicMock()
    mock_record = MagicMock()
    mock_record.data.return_value = {"filename": "file.csv"}
    mock_result.records = [mock_record]
    monkeypatch.setattr(graph.driver, "execute_query", lambda *a, **kw: mock_result)
    result = graph.fetch_filename_from_neo4j("metric")
    assert result == "file.csv"

def test_get_data_from_training_data(monkeypatch):
    df = pd.DataFrame({"filename": ["a", "b"], "other": [1, 2]})
    monkeypatch.setattr(pd, "read_json", lambda *a, **k: df)
    result = graph.get_data_from_training_data("a")
    assert not result.empty
    assert (result["filename"] == "a").all()

def test_domain_state_tracker():
    from langchain.schema.messages import HumanMessage
    messages = [HumanMessage(content="hi")]
    result = graph.domain_state_tracker(messages)
    assert result[0].content
    assert isinstance(result, list)

def test_call_llm(monkeypatch):
    monkeypatch.setattr(graph, "llm_with_tool", MagicMock(invoke=lambda x: MagicMock(content="ok")))
    state = {"messages": ["msg"]}
    result = graph.call_llm(state)
    assert "messages" in result

def test_process_user_question():
    state = {"messages": [{"tool_calls": [{"args": {
        "question": "q", "operation": "op", "timeframe": ["tf"], "metric": ["m"]
    }, "id": "id1"}]}]}
    result = graph.process_user_question(state)
    assert "messages" in result

def test_process_filename(monkeypatch):
    state = {"messages": [{"tool_calls": [{"args": {"metricname": "m"}, "id": "id1"}]}]}
    monkeypatch.setattr(graph.fetch_filename_from_neo4j, "run", lambda x: "file.csv")
    result = graph.process_filename(state)
    assert "messages" in result

def test_formula_agent_node(monkeypatch):
    monkeypatch.setattr(graph, "llm", MagicMock(invoke=lambda x: MagicMock(content="formula")))
    state = {"messages": [MagicMock(content="metric")]}
    result = graph.formula_agent_node(state)
    assert "messages" in result

def test_calculator_agent_node(monkeypatch):
    monkeypatch.setattr(graph, "llm", MagicMock(invoke=lambda x: MagicMock(content="calc")))
    state = {"messages": [MagicMock(content="formula")]}
    result = graph.calculator_agent_node(state)
    assert "messages" in result