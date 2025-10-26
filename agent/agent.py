from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pydantic import BaseModel
from typing import Literal
from langgraph.prebuilt import ToolNode, tools_condition

from agent.utils.prompt import CHAT_SYSTEM_PROMPT, WELCOME_MESSAGE, ROUTER_PROMPT
from agent.utils.state import StateSchema
from agent.utils.tools import TOOLS_CHAT




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
    
    
    graph = StateGraph(state_schema=StateSchema)


    def welcome_node(state: StateSchema) -> StateSchema:

        return {
            "messages": [AIMessage(content=WELCOME_MESSAGE)]
        }
  

    def router_node(state: StateSchema) -> str:

        class RouterOutput(BaseModel):
            route: str

        system_message = SystemMessage(content=ROUTER_PROMPT)

        response = llm.with_structured_output(RouterOutput).invoke([system_message, *state["messages"]])

        state["route"] = response.route

        return state


    def chat_node(state: StateSchema) -> StateSchema:

        system_message = SystemMessage(content=CHAT_SYSTEM_PROMPT)

        llm_with_tools = llm.bind_tools(tools=TOOLS_CHAT)

        response =  llm_with_tools.invoke([system_message, *state["messages"]])

        return {
            "messages": [response]
        }
    
    def guide_node(state: StateSchema) -> StateSchema:


        return state
    
    graph.add_node(welcome_node, name="welcome_node")
    graph.add_node(chat_node, name="chat_node")
    graph.add_node(ToolNode(tools=TOOLS_CHAT, name="tools_chat"), name="tools_chat")
    graph.add_node(router_node, name="router_node")
    graph.add_node(guide_node, name="guide_node")


   
 
    graph.add_edge("welcome_node", END)
    graph.add_edge("chat_node", END)
    graph.add_edge("guide_node", END)
    graph.add_edge("tools_chat", "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools_chat", "__end__": "__end__"})


    def welcome_condition(state:  StateSchema) -> Literal["router_node", "welcome_node"]:

        if len(state["messages"]) == 1:
            return "welcome_node"
        else:
            return "router_node"

    graph.add_conditional_edges(START, welcome_condition)

    def route_condition(state:  StateSchema) -> Literal["chat_node", "guide_node"]:

        if state.get("route") == "chat_node":
            return "chat_node"
        else:
            return "guide_node"
        
    graph.add_conditional_edges("router_node", route_condition)


    saver = None #InMemorySaver()

    compiled_graph = graph.compile(checkpointer=saver if saver else None)


    return compiled_graph


   
graph = create_agent_graph()

    