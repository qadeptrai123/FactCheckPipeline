from pydantic import BaseModel, Field

class PipelineRequest(BaseModel):
    query: str = Field(..., description="The user query to process")
    refine_model: str = Field(default="gpt-3.5-turbo", description="Model to use for query refinement")
    retrieval_method: str = Field(default="vector", description="Method to use for document retrieval")
    inference_model: str = Field(default="gpt-4", description="Model to use for final inference")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
