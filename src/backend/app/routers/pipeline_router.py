from fastapi import APIRouter, Depends, HTTPException
from app.schemas.request import PipelineRequest
from app.schemas.response import PipelineResponse
from app.services.pipeline_service import PipelineService

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])

def get_pipeline_service() -> PipelineService:
    return PipelineService()

@router.post("/execute", response_model=PipelineResponse)
async def execute(
    request: PipelineRequest,
    service: PipelineService = Depends(get_pipeline_service)
):
    try:
        response = service.execute_pipeline(request)
        if response.status == "error":
            raise HTTPException(status_code=500, detail=response.final_answer)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
