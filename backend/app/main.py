"""
Main FastAPI Application Entrypoint for AcousticIQ.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.database.session import Base, engine
from backend.app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on startup
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Acoustic Deep Learning Based Predictive Maintenance System for Early Detection of Industrial Machine Failures",
    lifespan=lifespan,
)

# Explicit CORS policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global unhandled error handler to prevent internal tracebacks leak
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred while processing the acoustic request."},
    )

# Mount API routes
app.include_router(router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def root():
    return {
        "product": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
        "api_v1": settings.API_V1_PREFIX,
        "mission": "Industrial Acoustic Anomaly Detection for Predictive Maintenance",
    }
