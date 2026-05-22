from typing import Any, Literal
from pydantic import BaseModel, Field


JobStatus = Literal["queued", "running", "done", "error"]
PhaseStatus = Literal["pending", "running", "done", "error"]


class PipelinePhase(BaseModel):
    status: PhaseStatus = "pending"
    duration_seconds: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PipelineResponse(BaseModel):
    job_id: str
    status: JobStatus
    claim_text: str
    phases: dict[str, PipelinePhase]
    result: dict[str, Any] | None = None
    error: str | None = None


class PipelineStartResponse(BaseModel):
    job_id: str
    status: JobStatus

