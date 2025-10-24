from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
import os

MODEL_NAME = "gemini-2.5-flash"

def create_agent_graph():

    llm =  ChatGoogleGenerativeAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model=MODEL_NAME,
            temperature=0,
            max_tokens=2048,
            timeout=None,
            max_retries=2,            
        )
    
    graph = StateGraph()




    return

