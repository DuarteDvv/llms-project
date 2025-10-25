from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
import os


from agent.utils.prompt import SYSTEM_PROMPT
from agent.utils.state import StateSchema




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

    def chat_node(state: StateSchema) -> StateSchema:

        system_message = SystemMessage(content=SYSTEM_PROMPT)

        response =  llm.invoke([system_message, *state["messages"]])
        
        return {
            "messages": [response]
        }
    
    def end_node(state: StateSchema) -> StateSchema:
        return state
    

    graph.add_node(chat_node, name="chat_node")
    graph.add_node(end_node, name="end_node")


    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", "end_node")
    graph.add_edge("end_node", END)

    saver = None #InMemorySaver()

    compiled_graph = graph.compile(checkpointer=saver if saver else None)


    return compiled_graph


   
graph = create_agent_graph()

    