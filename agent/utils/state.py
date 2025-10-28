from typing import Annotated, List, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph.message import add_messages

class StateSchema(TypedDict):

    messages: Annotated[list, add_messages]
    route: Optional[str]

    user_data: Optional[Dict[str, Any]]
    debug: Optional[Any]
    confirmation: Optional[bool]
    



    