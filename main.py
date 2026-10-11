from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router


app = FastAPI(
    title="Data Sentinel API",
    description="AI Data Quality and ML Readiness Platform",
    version="1.0.0"
)


app = FastAPI(
    title="Data Sentinel API",
    description="AI Data Quality and ML Readiness Platform",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


app.include_router(
    router,
    prefix="/api"
)


@app.get("/health")
def root_health():
    return {"status": "success", "message": "Data Sentinel API is running"}

app.mount(
    "/",
    StaticFiles(
        directory="frontend",
        html=True
    ),
    name="frontend"
)