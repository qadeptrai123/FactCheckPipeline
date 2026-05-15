from pydantic import BaseModel
from typing import List, Union, Dict, Any

class PipelineResponse(BaseModel):
    original_query: str
    refined_query: str
    retrieved_context: List[Union[str, Dict[str, Any]]]
    final_answer: str
    status: str
