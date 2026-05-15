from app.schemas.request import PipelineRequest
from app.schemas.response import PipelineResponse
from app.components.refiner import QueryRefiner
from app.components.retriever import DocumentRetriever
from app.components.inferencer import LLMInferencer

class PipelineService:
    def __init__(self):
        self.refiner = QueryRefiner()
        self.retriever = DocumentRetriever()
        self.inferencer = LLMInferencer()

    def execute_pipeline(self, request: PipelineRequest) -> PipelineResponse:
        try:
            # Stage 1: Query Refinement
            refined_query = self.refiner.refine(
                query=request.query,
                model=request.refine_model
            )

            # Stage 2: Document Retrieval
            retrieved_context = self.retriever.retrieve(
                query=refined_query,
                method=request.retrieval_method,
                top_k=request.top_k
            )

            # Stage 3: LLM Inference
            final_answer = self.inferencer.generate(
                query=refined_query,
                context=retrieved_context,
                model=request.inference_model
            )

            return PipelineResponse(
                original_query=request.query,
                refined_query=refined_query,
                retrieved_context=retrieved_context,
                final_answer=final_answer,
                status="success"
            )
        except Exception as e:
            return PipelineResponse(
                original_query=request.query,
                refined_query="",
                retrieved_context=[],
                final_answer=f"Error: {str(e)}",
                status="error"
            )
