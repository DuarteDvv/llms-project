from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pydantic import BaseModel
from typing import Literal
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanInterruptConfig, ActionRequest
from agent.utils.prompt import CHAT_SYSTEM_PROMPT, WELCOME_MESSAGE, ROUTER_PROMPT, GUIDE_SYSTEM_PROMPT
from agent.utils.state import StateSchema
from agent.utils.tools import TOOLS_CHAT

interruptConfig = HumanInterruptConfig(
    allow_ignore=True,    # permite ignorar a interrupção
    allow_respond=True,   # permite feedback em texto
    allow_edit=False,     # não permite edição
    allow_accept=False     # não permite aceitação direta
)

MODEL_NAME = "gemini-2.5-flash-lite"

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

        state["confirmation"] = False

        return {
            "messages": [AIMessage(content=WELCOME_MESSAGE)]
        }
  

    def router_node(state: StateSchema) -> str:

        class RouterOutput(BaseModel):
            route: str

        system_message = SystemMessage(content=ROUTER_PROMPT)

        response = llm.with_structured_output(RouterOutput).invoke([system_message, *state["messages"]])

        state["route"] = response.route

        if state["route"] not in ["chat_node", "guide_node"]:
            state["route"] = "chat_node"

        return state
    

    def chat_node(state: StateSchema) -> StateSchema:

        system_prompt = SystemMessage(content=CHAT_SYSTEM_PROMPT)

        response =  llm.bind_tools(tools=TOOLS_CHAT).invoke([system_prompt, *state["messages"]])

        return {
            "messages": [response],
        }
    
    
    def guide_node(state: StateSchema) -> StateSchema:

        return {
            "messages": [AIMessage(content="Antes de prosseguirmos, gostaria de fazer algumas perguntas para personalizar melhor o guia para você.")]
        }
    
    def ask_name(state: StateSchema) -> StateSchema:

        question = "Qual é seu nome?"

        request = HumanInterrupt(
            action_request=ActionRequest(
                action=question,
                args={}
            ),
            config=interruptConfig
        )

        answer = interrupt([request])[0]

        if answer["type"] == "ignore":
            answer = "Não informado"
        else:
            answer = answer["args"]

        user_data = state.get("user_data", {})       
        user_data["nome"] = answer
        state["user_data"] = user_data

        return state
    
    def ask_age(state: StateSchema) -> StateSchema:

        question = "Qual é sua idade?"

        request = HumanInterrupt(
            action_request=ActionRequest(
                action=question,
                args={}
            ),
            config=interruptConfig
        )

        while True:
            answer = interrupt([request])[0]

            if answer["type"] == "ignore":
                answer = "Não informado"
                break
            else:
                answer = int(answer["args"])

            if not isinstance(answer, int) or answer < 0:
                question = "Por favor, insira uma idade válida. Qual é sua idade?"

                request = HumanInterrupt(   
                    action_request=ActionRequest(
                        action=question,
                        args={}
                    ),
                    config=interruptConfig
                )
                continue
            else:               
                break


        user_data = state.get("user_data", {})       
        user_data["idade"] = answer
        state["user_data"] = user_data

        return state
    
    def ask_period(state: StateSchema) -> StateSchema:

        question = "Há quantos meses você não menstrua? Caso, já tenha passado por menopausa, por favor, informe quantos meses se passaram desde então."

        request = HumanInterrupt(
            action_request=ActionRequest(
                action=question,
                args={}
            ),
            config=interruptConfig
        )

        while True:
            answer = interrupt([request])[0]

            if answer["type"] == "ignore":
                answer = "Não informado"
                break
            else:
                answer = int(answer["args"])

            if not isinstance(answer, int) or answer < 0:
                question = "Por favor, insira um número válido de meses. Há quantos meses ?"

                request = HumanInterrupt(   
                    action_request=ActionRequest(
                        action=question,
                        args={}
                    ),
                    config=interruptConfig
                )
                continue
            else:               
                break


        user_data = state.get("user_data", {})       
        user_data["tempo_menopausa"] = answer
        state["user_data"] = user_data

        return state
    
    def ask_sintomas(state: StateSchema) -> StateSchema:

        question = "Descreva em detalhes como você tem se sentido ultimamente. Quais sintomas você está enfrentando que te preocupam?"

        request = HumanInterrupt(
            action_request=ActionRequest(
                action=question,
                args={}
            ),
            config=interruptConfig
        )

        answer = interrupt([request])[0]

        if answer["type"] == "ignore":
            answer = "Não informado"
        else:
            answer = answer["args"]

        user_data = state.get("user_data", {})       
        user_data["sintomas"] = answer
        state["user_data"] = user_data

        return state
    
    def ask_alimentacao(state: StateSchema) -> StateSchema:

        question = "Você tem se alimentado bem ultimamente? Quais alimentos você tem consumido com mais frequência?"

        request = HumanInterrupt(
            action_request=ActionRequest(
                action=question,
                args={}
            ),
            config=interruptConfig
        )

        answer = interrupt([request])[0]

        if answer["type"] == "ignore":
            answer = "Não informado"
        else:
            answer = answer["args"]

        user_data = state.get("user_data", {})       
        user_data["alimentacao"] = answer
        state["user_data"] = user_data

        return state

    def show_user_data_node(state: StateSchema) -> StateSchema:
        user_data = state.get("user_data", {}) or {}

        if not user_data:
            content = (
                "Ainda não recebi informações suas. Quando estiver pronto, posso fazer as perguntas novamente."
            )
        else:
            header = "Obrigado por fornecer essas informações. Aqui está um resumo dos dados que você compartilhou:\n"
            sep = "────────────────────────────────────────\n"

            lines = [header, sep]

            for key, value in user_data.items():
                # torna a chave mais legível: 'tempo_menopausa' -> 'Tempo menopausa'
                pretty_key = key.replace("_", " ").capitalize()

                # formata valores compostos (por exemplo, dicts) de forma compacta
                if isinstance(value, dict):
                    val = ", ".join(f"{k}: {v}" for k, v in value.items())
                else:
                    val = str(value)

                lines.append(f"• {pretty_key}: {val}\n")

            lines.append(sep)
            lines.append("Se quiser alterar algum item, clique em ignorar para recomeçar.")

            content = "\n".join(lines)

        return {"messages": [AIMessage(content=content)]}

        

    def ask_confirmation(state: StateSchema) -> StateSchema:

        question = "Voce confirma que essas informações estão corretas e completas para prosseguirmos com o guia?"

        request = HumanInterrupt(
            action_request=ActionRequest(
                action=question,
                args={}
            ),
            config=HumanInterruptConfig(
                allow_ignore=True,
                allow_respond=False,
                allow_edit=False,
                allow_accept=True
            )
        )

        answer = interrupt([request])[0]

        if answer["type"] == "ignore":
            answer = False
        else:
            answer = True

        state["confirmation"] = answer

        return state
    
    def generate_guide(state: StateSchema) -> StateSchema:

        user_data = state.get("user_data", {}) or {}

        system_message = SystemMessage(content=GUIDE_SYSTEM_PROMPT)

        prompt_parts = [
            "Com base nas seguintes informações do usuário, gere um guia estruturado para discutir com um médico sobre menopausa:\n"
        ]

        for key, value in user_data.items():
            pretty_key = key.replace("_", " ").capitalize()
            prompt_parts.append(f"{pretty_key}: {value}\n")

        prompt_parts.append(
            "\nO guia deve incluir pontos principais, perguntas a serem feitas ao médico e quaisquer preocupações relevantes."
        )

        user_message = HumanMessage(content="".join(prompt_parts))

        response = llm.invoke([system_message, user_message])

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools=TOOLS_CHAT, name="tools_chat")
    
    graph.add_node(welcome_node, name="welcome_node")
    graph.add_node(chat_node, name="chat_node")
    graph.add_node(tool_node, name="tools_chat")
    graph.add_node(router_node, name="router_node")
    graph.add_node(guide_node, name="guide_node")
    graph.add_node(ask_age, name="ask_age")
    graph.add_node(ask_name, name="ask_name")
    graph.add_node(ask_period, name="ask_period")
    graph.add_node(ask_sintomas, name="ask_sintomas")
    graph.add_node(ask_alimentacao, name="ask_alimentacao")
    graph.add_node(show_user_data_node, name="show_user_data_node")
    graph.add_node(ask_confirmation, name="ask_confirmation")
    graph.add_node(generate_guide, name="generate_guide")



   
 
    graph.add_edge("welcome_node", END)
    graph.add_edge("chat_node", END)
    graph.add_edge("guide_node", "ask_name")
    graph.add_edge("ask_name", "ask_age")
    graph.add_edge("ask_age", "ask_period")
    graph.add_edge("ask_period", "ask_sintomas")
    graph.add_edge("ask_sintomas", "ask_alimentacao")
    graph.add_edge("ask_alimentacao", "show_user_data_node")
    graph.add_edge("show_user_data_node", "ask_confirmation")
    graph.add_edge("tools_chat", "chat_node")
    graph.add_edge("generate_guide", END)

    graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools_chat", "__end__": "__end__"})


    def data_condition(state:  StateSchema) -> Literal["ask_name", "generate_guide"]:

        if state.get("confirmation"):
            return "generate_guide"
        else:
            return "ask_name"

    graph.add_conditional_edges("ask_confirmation", data_condition)

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

    