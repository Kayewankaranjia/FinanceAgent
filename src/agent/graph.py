"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict
from langgraph.store.memory import InMemoryStore

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import pandas as pd

from typing import List, Literal, Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END,MessagesState
from langgraph.types import Command
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import uuid
from langchain.schema.messages import HumanMessage,SystemMessage , AIMessage, ToolMessage

from langchain_openai import OpenAIEmbeddings
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt import create_react_agent
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from prompts import PromptTemplates
import json
from neo4j import GraphDatabase
import re
import ast
prompt_templates = PromptTemplates()

load_dotenv(find_dotenv()) # read local .env file

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "pass@123")

        # Connect to Neo4j database
driver = GraphDatabase.driver(uri=os.getenv("Neo4j_URL"), auth=(os.getenv("Neo4j_USERNAME"), os.getenv("Neo4j_PASSWORD")))
openai_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))



class State(TypedDict):
    messages: List[add_messages]

def embed(texts: list[str]) -> list[list[float]]:
    # Replace with an actual embedding function or LangChain embeddings object
    return [[1.0, 2.0] * len(texts)]


store = InMemoryStore(index={"embed": embed, "dims": 2})


class UserQuestionModel(BaseModel):
    question: str
    operation: str
    timeframe: list
    metric: list

@tool
def user_question_tool(question: str, operation: str, timeframe: list, metric: list):
    """Tool for handling user questions."""
    return {
        "question": question,
        "operation": operation,
        "timeframe": timeframe,
        "metric": metric,
    }

@tool
def fetch_filename_from_neo4j(metricname:str ) -> str:
    """ Fetches the filename from Neo4j database based on the metric name."""
    print("metricname in tool", metricname)
    query_embedding = openai_embeddings.embed_query(metricname)

#     query = """
# MATCH (node:Cell)
# WITH node, vector.similarity.cosine($QueryEmbedding, node.cell_embedding) AS score
# RETURN node.id
# ORDER BY score DESCENDING
# LIMIT 1;
#         """

    query = """ 
    MATCH (cell:Cell)
    WITH cell, vector.similarity.cosine($QueryEmbedding, cell.cell_embedding) AS score
    ORDER BY score DESC
    LIMIT 1
    MATCH (cell)<-[:HAS_CELL]-(row:TableRow)<-[:HAS_ROW]-(table:Table)<-[:HAS_TABLE]-(doc:Document)
    RETURN doc.filename AS filename"""
    result = driver.execute_query(query,  QueryEmbedding =  query_embedding)
    print(f"Result from Neo4j: {result}")
    try:
        records = result.records if hasattr(result, 'records') else result
        # Neo4j returns a list of Record objects; access the 'filename' field
        if records and hasattr(records[0], 'data'):
            filename = records[0].data().get('filename', "Filename not found")
        elif records and 'filename' in records[0]:
            filename = records[0]['filename']
        else:
            filename = "Filename not found"
    except Exception as e:
        filename = f"Error retrieving filename: {str(e)}"
    

    return filename

@tool
def basic_calculator_tool(expression: str):
    """Evaluate a simple math expression and return the result."""
    try:
        result = eval(expression)
        return float(result)  # <-- Return just the result
    except Exception as e:
        return f"Error evaluating expression: {e}"  

def get_data_from_training_data(filename: str) -> pd.DataFrame:
    """Fetches the document from training data with the given filename."""
   
    print("filename file", filename)
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.json")
    df = pd.read_json(data_path)

    # Drop duplicates based only on hashable columns (exclude columns with lists/dicts)
    df_unique = df.drop_duplicates(subset=['filename']) 

    # filter for the row(s) where filename equals your target
    filename_to_find = filename.strip()
    print("filemae", filename_to_find)
    result_df = df_unique[df_unique['filename'] == filename_to_find]
    print("resultdf", result_df)
    return result_df

llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

llm_with_tool = llm.bind_tools([user_question_tool])





def domain_state_tracker(messages: List[HumanMessage]) -> List[SystemMessage]:
    return [SystemMessage(content=prompt_templates.prompt_system_task.format())] + messages

def call_llm(state: State):
    messages = domain_state_tracker(state["messages"])
    
    response = llm_with_tool.invoke(messages)

    # userquestion = UserQuestion(response.content)
    # print(f"User question: {userquestion}")
    return {"messages": [response]}

def process_user_question(state: State):
     tool_args = state["messages"][-1].tool_calls[0]["args"]
     userquestion = UserQuestionModel(
        question=tool_args["question"],
        operation=tool_args["operation"],
        timeframe=tool_args["timeframe"],
        metric=tool_args["metric"],
    )
     
     return {
        "messages": [
            ToolMessage(
                content="User question received!",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                artifact=userquestion,
            )
        ]
    }

def get_filename(state: State):

    """
    Get the relevent filename 
    """
    messages = domain_state_tracker(state["messages"])

    # Extract the last HumanMessage content (assumed to be a JSON string)
    last_message = state["messages"][-1]
    print(f"Last message: {last_message}")
    user_data = last_message.artifact
    store.put(
        namespace="user_question",
        key="user_question",
        value=user_data.question
        
    )
    store.put(
        namespace="user_question",
        key="operation",
        value=user_data.operation
    )
    store.put(
        namespace="user_question",
        key="timeframe",
        value=user_data.timeframe
    )
    store.put(
        namespace="user_question",
        key="metric",
        value=user_data.metric
    )

    messages = [SystemMessage(content=prompt_templates.file_name_prompt.format(metric=user_data.metric))] 
    
    llm_with_tool = llm.bind_tools([fetch_filename_from_neo4j])

    result = llm_with_tool.invoke(messages)
    print("IN TEResult", result)
    return {"messages": [result]}

def process_filename(state: State):
    tool_args = state["messages"][-1].tool_calls[0]["args"]
    timeframe = store.get( namespace="user_question",
        key="timeframe")
    input = tool_args["metricname"] 
    print(f"in Tool args: {tool_args["metricname"]}")
    last_tool_result = fetch_filename_from_neo4j(tool_input = input)
    print(f"Last tool result(filename): {last_tool_result}")
    user_data  = tool_args
    user_data["filename"] = last_tool_result
    store.put(
        namespace="user_question",
        key="filename",
        value=last_tool_result
    )
    
    return {
        "messages": [
            ToolMessage(
                content=user_data,
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            ),
        ]
    }

   

def generate_components(state):
    messages = domain_state_tracker(state["messages"])
    # Extract the last HumanMessage content (assumed to be a JSON string)
    last_message = state["messages"][-1]
    userquestion = store.get(
        namespace="user_question",
        key="user_question"
    )
    
    messages = [SystemMessage(content=prompt_templates.components_prompt.format(question=userquestion))]
    
    result = llm.invoke(messages)

    print("Result form LLMS :" , result.content)

    # Clean up the result.content string to extract the JSON object

    # Remove python code block markers and whitespace
    content_str = str(result.content)
    content_str = re.sub(r"^```python\s*|```$", "", content_str.strip(), flags=re.MULTILINE)
    content_str = content_str.strip()

    # Try to parse as dict
    try:
        result_dict = ast.literal_eval(content_str)
    except Exception:
        result_dict = {}

    print(result_dict)

   
    store.put(
        namespace="user_question",
        key="Operation to be Performed",
        value=str(result_dict['OperationToBePerformed']))
    
    store.put(
        namespace="user_question",
        key="RequiredTimeFrames",
        value=str(result_dict['RequiredTimeFrames'])
    )
    
   
    return{
        "messages": [{"componenets" : result_dict}]
        
    }


def fetch_data(state: State):
    
    components = state["messages"][-1]
    filename = store.get(
        namespace="user_question",
        key="filename"
    ).value
     
    data = get_data_from_training_data(filename=filename)
    data = data.to_string()

    

    print("Big data", data)
    messages = prompt_templates.retrive_prompt.format(components=components, data=data)
    result = llm.invoke(messages)

    print("Fetch Data result", result)
   
    data = result

    return {"messages": [data]}

def formula_agent_node(state: State):
    """
    Fetches formulas based on the topic identified in the previous step.Return a single or multiple formulas whih can be used to calculate the financial metric.
    """
   
    messages = state["messages"]
    messages.append(SystemMessage(content=prompt_templates.math_formula_prompt.format(response=state["messages"][-1].content)))
    messages.append(HumanMessage(content="What is the formula for the metric?"))
    # llm_with_tool = llm.bind_tools([create_math_formula])
    result = llm.invoke(messages)  
    return {"messages": [result]}

def calculator_agent_node(state: State):
    """
    Perfoms the calculation based on the formula created in the previous step.
    """
    messages = domain_state_tracker(state["messages"])
    messages.append(HumanMessage(content="Calculate the financial metric using the formula created in the previous step for the opration mentioned"))
    messages.append(SystemMessage(content="You are a calculator agent. You will be given a math expression to evaluate. Pass the expression to the calculator tool and return the result."))
    
    llm_with_tool = llm.bind_tools([basic_calculator_tool])

    result = llm_with_tool.invoke(messages)
    print("cal agent", result)
    return {"messages": [result]}

def process_calculatortool(state: State):
    tool_args = state["messages"][-1].tool_calls[0]["args"]
    print(f"in Tool args: {tool_args["expression"]}")
    last_tool_result = basic_calculator_tool(tool_input = tool_args["expression"])
    print(f"Last tool result(filename): {last_tool_result}")
    store.put(
        namespace="user_question",
        key="calculated_result",
        value=last_tool_result
    )
    
    return {
        "messages": [
            ToolMessage(
                content={"calculated_result" : last_tool_result},
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            ),
        ]
    }

def call_model_to_printfinalmessage(state):
    calresult = state["messages"]
    userquestion = store.get(
        namespace="user_question",
        key="user_question"
    )
    messages = prompt_templates.final_message_prompt.format(userquestion=userquestion, answer=calresult)
    result = llm.invoke(messages)

    print("Fetch Data result", result)

    return {"messages": [result]}




workflow = StateGraph(State)
workflow.add_node("call_llm", call_llm)
workflow.add_edge(START, "call_llm")
workflow.add_node("process_user_question", process_user_question)
workflow.add_edge("call_llm", "process_user_question")
workflow.add_node("get_filename", get_filename)
workflow.add_edge("process_user_question", "get_filename")
workflow.add_node("process_filename", process_filename)
workflow.add_edge("get_filename", "process_filename")
workflow.add_node("generate_components", generate_components)
workflow.add_edge("process_filename", "generate_components")
workflow.add_node("fetch_data", fetch_data)
workflow.add_edge("generate_components", "fetch_data")
workflow.add_node("formula_agent", formula_agent_node)
workflow.add_edge("fetch_data", "formula_agent")
workflow.add_node("calculator_agent", calculator_agent_node)
workflow.add_edge("formula_agent", "calculator_agent")
workflow.add_node("process_calculatortool", process_calculatortool)
workflow.add_edge("calculator_agent", "process_calculatortool")
workflow.add_node("print_final_message", call_model_to_printfinalmessage)
workflow.add_edge("process_calculatortool", "print_final_message")
workflow.add_edge("print_final_message", END)
graph = workflow.compile()