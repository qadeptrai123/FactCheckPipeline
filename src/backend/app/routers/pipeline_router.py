from fastapi import APIRouter, Depends, HTTPException

from app.schemas.request import PipelineRequest
from app.schemas.response import PipelineResponse, PipelineStartResponse
from app.services.pipeline_service import PipelineService


router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])
pipeline_service = PipelineService()


def get_pipeline_service() -> PipelineService:
    return pipeline_service


@router.post("/execute", response_model=PipelineStartResponse)
async def execute(
    request: PipelineRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineStartResponse:
    job = service.start_pipeline(request)
    return PipelineStartResponse(job_id=job.job_id, status=job.status)


@router.get("/status/{job_id}", response_model=PipelineResponse)
async def status(
    job_id: str,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineResponse:
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Pipeline job not found")
    return job


@router.get("/debug/runtime")
async def runtime(
    service: PipelineService = Depends(get_pipeline_service),
) -> dict:
    return service.diagnostics()
