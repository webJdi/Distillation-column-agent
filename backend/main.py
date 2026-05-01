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
import asyncio
import queue

from backend.config import settings
from backend.api import prices, simulation, training, disturbance, ai_agent


async def _broadcast_queue_processor():
    """
    Background task that monitors the broadcast queue and sends messages
    to all connected WebSocket clients. Runs continuously during app lifetime.
    """
    from backend.api.training import _broadcast_queue, _broadcast_to_ws
    
    while True:
        try:
            # Non-blocking check for messages
            try:
                data = _broadcast_queue.get_nowait()
                await _broadcast_to_ws(data)
            except queue.Empty:
                # No message, wait a bit and check again
                await asyncio.sleep(0.05)
        except Exception as exc:
            logger.warning(f"Broadcast queue processor error: {exc}")
            await asyncio.sleep(0.1)


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

    # Auto-load the most recent notebook SAC checkpoint as the default inference model
    import glob
    nb_dir = os.path.join(settings.RL_CHECKPOINT_DIR, "notebook")
    sac_checkpoints = sorted(
        glob.glob(os.path.join(nb_dir, "notebook_SAC_*.zip")), reverse=True
    )
    if sac_checkpoints:
        from backend.api.training import agent_manager
        try:
            agent_manager.load_checkpoint(sac_checkpoints[0])
            logger.info(f"✅  Auto-loaded SAC checkpoint: {sac_checkpoints[0]}")
        except Exception as exc:
            logger.warning(f"⚠️  Could not auto-load checkpoint: {exc}")
    else:
        logger.info("ℹ️  No notebook SAC checkpoint found — inference unavailable until trained")

    # Start background broadcast processor
    processor_task = asyncio.create_task(_broadcast_queue_processor())
    logger.info("✅  Broadcast queue processor started")

    yield

    # Cleanup
    processor_task.cancel()
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
