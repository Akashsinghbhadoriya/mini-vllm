from pydantic import BaseModel
from typing import Literal, Optional

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 128
    stream: bool = False
    temperature: float = 1.0

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 128
    stream: bool = False
    temperature: float = 1.0

# --- Outbound (non-streaming) ---                                                                            
class UsageInfo(BaseModel):                                     
    prompt_tokens: int                                          
    completion_tokens: int                                      
    total_tokens: int                                           
                                                                
class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str = "stop"
                                                                
class CompletionResponse(BaseModel):
    id: str                                                     
    object: str = "text_completion"                   
    created: int                               
    model: str                                                 
    choices: list[CompletionChoice]
    usage: UsageInfo

class ChatCompletionMessage(BaseModel):                         
    role: str = "assistant"
    content: str                                                
                                                    
class ChatCompletionChoice(BaseModel):         
    message: ChatCompletionMessage                             
    index: int = 0
    finish_reason: str = "stop"
                                                                
class ChatCompletionResponse(BaseModel):
    id: str                                                     
    object: str = "chat.completion"                   
    created: int                               
    model: str                                                 
    choices: list[ChatCompletionChoice]
    usage: UsageInfo

# --- SSE streaming chunks ---                                  
class DeltaContent(BaseModel):                                  
    content: str = ""                                 
                                                
class StreamChoice(BaseModel):                                 
    delta: DeltaContent
    index: int = 0                                              
    finish_reason: Optional[str] = None
                                                                
class StreamChunk(BaseModel):                         
    id: str                                    
    object: str = "chat.completion.chunk"                      
    created: int
    model: str
    choices: list[StreamChoice]