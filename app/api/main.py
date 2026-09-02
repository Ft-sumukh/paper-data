import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.database.connection import init_db
from app.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Financial Market Risk & Portfolio Research API...")
    init_db()
    yield
    logger.info("Shutting down API service...")

app = FastAPI(
    title="Financial Market Risk & Portfolio Optimization Research API",
    description="Quantitative research platform API for convex portfolio optimization, risk measurement, walk-forward backtesting, and market regime analysis.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "service": "Financial Market Risk & Portfolio Optimization Platform",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
