from pydantic import BaseModel
from typing import List, Optional

class ChatRecord(BaseModel):
    role: str
    content: str

class ConversationControl(BaseModel):
    callWithVectorStore: bool = False
    callWithDecisionService: bool = False
    query: str = ""
    locale: str= "en"
    type: str = "chat"
    chat_history: List[ChatRecord] = []

class ResponseChoice(BaseModel):
    choice: str = None

class ResponseControl(BaseModel):
    message: Optional[str] = ''
    status: int = 200
    type: str ="OpenQuestion"
    question: Optional[str] = ''
    question_type: Optional[str] = ''
    possibleResponse: Optional[List[ResponseChoice]] = None
    error: Optional[str] = ''