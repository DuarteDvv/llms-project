from langgraph.prebuilt import ToolNode, tools_condition
from langchain.messages import HumanMessage, AIMessage, SystemMessage
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

MODEL_NAME = "gemini-2.5-flash"

def create_agent_graph():

    llm =  ChatGoogleGenerativeAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model=MODEL_NAME,
        temperature=0,
        max_tokens=20000,
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
    
    def ask_email(state: StateSchema) -> StateSchema:

        question = "Qual é o seu email? (Usaremos para enviar o guia personalizado)"

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
        user_data["email"] = answer
        state["user_data"] = user_data

        return state
    
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

        answer = interrupt([request])[0]

        if answer["type"] == "ignore":
            answer = "Não informado"
        else:
            answer = answer["args"]

        user_data = state.get("user_data", {})
        user_data["idade"] = answer
        state["user_data"] = user_data

        return state
    
    def ask_ciclo_menstrual(state: StateSchema) -> StateSchema:

        question = "Como está o seu ciclo menstrual? (Quando foi sua última menstruação, ela tem sido regular em frequência e fluxo? Você já completou 12 meses consecutivos sem menstruar?)"

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
        user_data["ciclo_menstrual"] = answer
        state["user_data"] = user_data

        return state
    
    def ask_sintomas_fisicos(state: StateSchema) -> StateSchema:

        question = "Quais sintomas físicos novos ou incômodos você tem sentido? (Por exemplo: ondas de calor, suores noturnos, alterações no sono, cansaço, ressecamento vaginal, mudanças na libido, ganho de peso, queda de cabelo ou infecções urinárias?)"

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
        user_data["sintomas_fisicos"] = answer
        state["user_data"] = user_data

        return state
    
    def ask_saude_emocional(state: StateSchema) -> StateSchema:

        question = "Como você tem se sentido emocional e mentalmente? (Você notou flutuações de humor, ansiedade, irritabilidade, desânimo, ou dificuldade de memória e concentração?)"

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
        user_data["saude_emocional"] = answer
        state["user_data"] = user_data

        return state
    
    def ask_habitos_historico(state: StateSchema) -> StateSchema:

        question = "Como estão seus hábitos de saúde e histórico médico? (Incluindo medicamentos ou suplementos que você usa, seu histórico pessoal ou familiar de doenças crônicas, especialmente câncer de mama, sua rotina de alimentação, exercícios, consumo de álcool ou fumo.)"

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
        user_data["habitos_historico"] = answer
        state["user_data"] = user_data

        return state
    
    def ask_exames_tratamentos(state: StateSchema) -> StateSchema:

        question = "Quando você realizou seus últimos exames preventivos e quais tratamentos você gostaria de discutir? (Como Papanicolau, mamografia e densitometria óssea. Você já tentou algo para os sintomas ou tem interesse em discutir opções, como a terapia de reposição hormonal?)"

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
        user_data["exames_tratamentos"] = answer
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

        # Mapeamento das perguntas feitas ao usuário
        questions_map = {
            "email": "Qual é o seu email? (Usaremos para enviar o guia personalizado)",
            "nome": "Qual é seu nome?",
            "idade": "Qual é sua idade?",
            "ciclo_menstrual": "Como está o seu ciclo menstrual? (Quando foi sua última menstruação, ela tem sido regular em frequência e fluxo? Você já completou 12 meses consecutivos sem menstruar?)",
            "sintomas_fisicos": "Quais sintomas físicos novos ou incômodos você tem sentido? (Por exemplo: ondas de calor, suores noturnos, alterações no sono, cansaço, ressecamento vaginal, mudanças na libido, ganho de peso, queda de cabelo ou infecções urinárias?)",
            "saude_emocional": "Como você tem se sentido emocional e mentalmente? (Você notou flutuações de humor, ansiedade, irritabilidade, desânimo, ou dificuldade de memória e concentração?)",
            "habitos_historico": "Como estão seus hábitos de saúde e histórico médico? (Incluindo medicamentos ou suplementos que você usa, seu histórico pessoal ou familiar de doenças crônicas, especialmente câncer de mama, sua rotina de alimentação, exercícios, consumo de álcool ou fumo.)",
            "exames_tratamentos": "Quando você realizou seus últimos exames preventivos e quais tratamentos você gostaria de discutir? (Como Papanicolau, mamografia e densitometria óssea. Você já tentou algo para os sintomas ou tem interesse em discutir opções, como a terapia de reposição hormonal?)"
        }

        prompt_parts = [
            "Crie um guia personalizado de menopausa com base nas seguintes informações coletadas:\n\n"
        ]

        filtered_data = {k: v for k, v in user_data.items() if k != "guide"}

        if not filtered_data or len(filtered_data) == 0:
            # se não houver dados, criar um guia genérico
            prompt_parts.append("Informações do paciente: Dados não informados\n")
        else:
            prompt_parts.append("=== PERGUNTAS E RESPOSTAS DA PACIENTE ===\n\n")
            for key, value in filtered_data.items():
                # Adiciona a pergunta correspondente
                question = questions_map.get(key, key.replace("_", " ").capitalize())
                
                if value and value != "Não informado":
                    prompt_parts.append(f"PERGUNTA: {question}\n")
                    prompt_parts.append(f"RESPOSTA: {value}\n\n")

        prompt_parts.append(
            "\nGere o guia completo seguindo EXATAMENTE o formato especificado no system prompt, "
            "incluindo os marcadores [INICIO_GUIA] e [FIM_GUIA]. "
            "Use as perguntas e respostas acima como contexto para personalizar o guia de forma detalhada e relevante."
        )

        user_message = HumanMessage(content="".join(prompt_parts))

        try:
            print(f"[DEBUG] Gerando guia com dados: {filtered_data}")
            
            response = llm.invoke([system_message, user_message])
            
            if not response or not response.content:
                # Fallback se não houver conteúdo
                fallback_guide_content = (
                    "# Guia Personalizado para Consulta sobre Menopausa\n\n"
                    "## 📋 Informações da Paciente\n"
                    "Informações não fornecidas.\n\n"
                    "## 🔍 Resumo da Situação Atual\n"
                    "Este guia foi criado para ajudá-la a preparar sua consulta médica sobre menopausa.\n\n"
                    "## 🩺 Sintomas e Observações\n"
                    "- Sintomas não especificados\n\n"
                    "## ❓ Perguntas Importantes para o Médico\n"
                    "1. Quais são os sintomas mais comuns da menopausa?\n"
                    "2. Quais tratamentos estão disponíveis para mim?\n"
                    "3. Como posso melhorar minha qualidade de vida durante este período?\n"
                    "4. Existem mudanças no estilo de vida que você recomenda?\n"
                    "5. Quando devo retornar para acompanhamento?\n\n"
                    "## 💡 Recomendações de Bem-Estar\n"
                    "- Mantenha uma alimentação equilibrada rica em cálcio e vitamina D\n"
                    "- Pratique exercícios físicos regularmente\n"
                    "- Cuide da saúde mental e busque apoio quando necessário\n"
                    "- Mantenha-se hidratada\n\n"
                    "## 📌 Próximos Passos\n"
                    "- Anote qualquer sintoma novo antes da consulta\n"
                    "- Leve este guia impresso ou em formato digital\n"
                    "- Não hesite em fazer todas as suas perguntas ao médico\n\n"
                    "---\n"
                    "*Este guia foi gerado para auxiliar na preparação da sua consulta médica.*"
                )
                
                full_response = (
                    f"[INICIO_GUIA]\n{fallback_guide_content}\n[FIM_GUIA]\n\n"
                    "Pronto! Seu guia personalizado foi gerado com sucesso! 📋✨ "
                    "Gostaria que eu enviasse este guia para o seu email?"
                )
                
                response = AIMessage(content=full_response)
            
            content = response.content
            guide_content = content
            
            if "[INICIO_GUIA]" in content and "[FIM_GUIA]" in content:
                start_idx = content.find("[INICIO_GUIA]") + len("[INICIO_GUIA]")
                end_idx = content.find("[FIM_GUIA]")
                guide_content = content[start_idx:end_idx].strip()
            
            if "user_data" not in state:
                state["user_data"] = {}
            state["user_data"]["guide"] = guide_content
            
            print(f"[DEBUG] Guia salvo com {len(guide_content)} caracteres")

            return {
                "messages": [response],
                "user_data": state["user_data"]
            }
        
        except Exception as e:
            print(f"[ERROR] Erro ao gerar guia: {str(e)}")
           
            error_message = AIMessage(
                content=f"Desculpe, houve um problema ao gerar o guia. Por favor, tente novamente mais tarde. Se o problema persistir, entre em contato com o suporte."
            )
            return {
                "messages": [error_message],
                "user_data": state.get("user_data", {})
            }

    tool_node = ToolNode(tools=TOOLS_CHAT, name="tools_chat")
    
    graph.add_node(welcome_node, name="welcome_node")
    graph.add_node(chat_node, name="chat_node")
    graph.add_node(tool_node, name="tools_chat")
    graph.add_node(router_node, name="router_node")
    graph.add_node(guide_node, name="guide_node")
    graph.add_node(ask_email, name="ask_email")
    graph.add_node(ask_age, name="ask_age")
    graph.add_node(ask_name, name="ask_name")
    graph.add_node(ask_ciclo_menstrual, name="ask_ciclo_menstrual")
    graph.add_node(ask_sintomas_fisicos, name="ask_sintomas_fisicos")
    graph.add_node(ask_saude_emocional, name="ask_saude_emocional")
    graph.add_node(ask_habitos_historico, name="ask_habitos_historico")
    graph.add_node(ask_exames_tratamentos, name="ask_exames_tratamentos")
    graph.add_node(show_user_data_node, name="show_user_data_node")
    graph.add_node(ask_confirmation, name="ask_confirmation")
    graph.add_node(generate_guide, name="generate_guide")



   
 
    graph.add_edge("welcome_node", END)
    graph.add_edge("chat_node", END)
    graph.add_edge("guide_node", "ask_email")
    graph.add_edge("ask_email", "ask_name")
    graph.add_edge("ask_name", "ask_age")
    graph.add_edge("ask_age", "ask_ciclo_menstrual")
    graph.add_edge("ask_ciclo_menstrual", "ask_sintomas_fisicos")
    graph.add_edge("ask_sintomas_fisicos", "ask_saude_emocional")
    graph.add_edge("ask_saude_emocional", "ask_habitos_historico")
    graph.add_edge("ask_habitos_historico", "ask_exames_tratamentos")
    graph.add_edge("ask_exames_tratamentos", "show_user_data_node")
    graph.add_edge("show_user_data_node", "ask_confirmation")
    graph.add_edge("tools_chat", "chat_node")
    graph.add_edge("generate_guide", END)

    graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools_chat", "__end__": "__end__"})


    def data_condition(state:  StateSchema) -> Literal["ask_email", "generate_guide"]:

        if state.get("confirmation"):
            return "generate_guide"
        else:
            return "ask_email"

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


    compiled_graph = graph.compile()


    return compiled_graph


   
graph = create_agent_graph()

    