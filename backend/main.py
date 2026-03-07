"""
CDU Optimizer — FastAPI Backend

Main application entry point that mounts all API routers,
configures CORS for the React frontend, and serves
the WebSocket endpoint for real-time training progress.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from loguru import logger
import os

from backend.config import settings
from backend.api import prices, simulation, training, disturbance, ai_agent


# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info(f"🚀  {settings.APP_NAME} v{settings.APP_VERSION} starting")
    logger.info(f"   DWSIM path : {settings.DWSIM_PATH}")
    logger.info(f"   Flowsheet  : {settings.FLOWSHEET_PATH}")

    # Ensure data directories exist
    os.makedirs(settings.RL_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    yield

    # Cleanup
    logger.info("Shutting down …")
    try:
        from backend.api.simulation import _bridge
        if _bridge:
            _bridge.close()
    except Exception:
        pass


# App

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Deep RL-powered Crude Distillation Unit optimizer.\n\n"
        "Components:\n"
        "- **Prices**: Manage product prices & market scenarios\n"
        "- **Simulation**: DWSIM CDU model interaction\n"
        "- **Training**: RL agent training & inference\n"
        "- **Disturbance**: Feed disturbance analysis\n"
        "- **AI Agent**: Explanations, reports, Q&A"
    ),
    lifespan=lifespan,
)

# CORS — allow the React frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(prices.router)
app.include_router(simulation.router)
app.include_router(training.router)
app.include_router(disturbance.router)
app.include_router(ai_agent.router)


# Health check

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
