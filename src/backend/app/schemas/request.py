from pydantic import BaseModel, Field


class PipelineRequest(BaseModel):
    claim_text: str = Field(..., min_length=1, description="Vietnamese claim to fact-check")
    claim_image: str | None = Field(default=None, description="Optional image path")

