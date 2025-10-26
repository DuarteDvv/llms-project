from langchain_core.tools import tool
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"



llm =  ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model=MODEL_NAME,
    temperature=0,
    max_tokens=2048,
    timeout=None,
    max_retries=2,            
)

class Subqueries(BaseModel):
    subquery1: str
    subquery2: str
    subquery3: str

@tool
def retrieve_information(query: str) -> str:
    """Retorna documentos informacoes confiaveis e relevantes sobre aspectos da menopausa.
    Esta ferramenta é útil para obter informações detalhadas sobre sintomas, tratamentos,
    impacto na saúde mental, dicas de estilo de vida e outros tópicos relacionados à saúde da mulher durante a menopausa.
    Args:
        query (str): A consulta sobre a qual recuperar informações.
    Returns:
        str: Documentos informativos relevantes sobre a consulta.

    """

    response = llm.with_structured_output(Subqueries).invoke([
        HumanMessage(content=f"Divida a seguinte consulta em três subconsultas distintas e relevantes que complementem a original: {query}")
    ])


    return "\n".join([response.subquery1, response.subquery2, response.subquery3])


#retrieve.invoke("Quais são as opções de tratamento para sintomas de menopausa e como elas afetam a saúde óssea?")


TOOLS_CHAT = [retrieve_information]