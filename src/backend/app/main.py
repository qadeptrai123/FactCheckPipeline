from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import pipeline_router

app = FastAPI(title="AI Pipeline API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(pipeline_router.router)

@app.on_event("startup")
def preload_pipeline_models():
    pipeline_router.pipeline_service.preload_components()

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "AI Pipeline API is running"}
