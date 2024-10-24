from pydantic import BaseModel
from typing import List, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage
#from app.doc_processing.filters import unwanted_titles_list_default, unwanted_categories_default

class ProcessDocConfigAPI(BaseModel):
    tag: str
    unwanted_titles_list: Optional[List[str]] = None #unwanted_titles_list_default
    unwanted_categories_list: Optional[List[str]] = None #unwanted_categories_default
    local: bool = True
    filepath_id: Optional[str] = None
    filepath: Optional[str] = None
    url: Optional[str] = None
    situate_context: bool = False
    late_chunking: bool = False

class ChatConfigAPI(BaseModel):
    tag: str
    expand_by_answer: bool = False
    expand_by_mult_queries: bool = False
    reranking: bool = True
    k: int = 10
    llm_choice: str = "groq"
    use_bm25: bool = False
    history_awareness: bool = False
    conversation_history: Optional[List[Union[HumanMessage, AIMessage]]] = None