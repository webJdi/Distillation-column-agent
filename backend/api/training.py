"""API routes for RL training management + WebSocket for live progress."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Optional
import asyncio
import json

from backend.models.schemas import (
    TrainingConfig,
    TrainingProgress,
    TrainingStatus,
    OptimizationRequest,
    OptimizationResult,
    ColumnAction,
)
from backend.core.rl_agent import RLAgentManager, ProgressCallback
from backend.core.rl_environment import CDUEnvironment, ACTION_KEYS, PRODUCT_KEYS, DEFAULT_PRICES
from backend.services.firebase_service import FirebaseService
from loguru import logger
import numpy as np

router = APIRouter(prefix="/api/training", tags=["Training"])

# Singletons
agent_manager = RLAgentManager()
firebase = FirebaseService()

# Connected WebSocket clients for live progress
_ws_clients: set[WebSocket] = set()


async def _broadcast_to_ws(data: dict):
    """Send training progress to all connected WebSocket clients."""
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    _ws_clients -= dead


def _sync_broadcast(data: dict):
    """Sync wrapper for the async broadcast (called from training thread)."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_broadcast_to_ws(data))
        else:
            loop.run_until_complete(_broadcast_to_ws(data))
    except RuntimeError:
        pass  # no event loop in training thread — that's OK, we'll poll


# ── WebSocket endpoint ──────────────────────────────────────────────────────

@router.websocket("/ws")
async def training_ws(websocket: WebSocket):
    """WebSocket endpoint for real-time training progress updates."""
    await websocket.accept()
    _ws_clients.add(websocket)
    logger.info(f"WS client connected ({len(_ws_clients)} total)")
    try:
        while True:
            # Keep connection alive + accept commands
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif msg.get("type") == "get_progress":
                progress = agent_manager.latest_progress
                if progress:
                    await websocket.send_json(progress)
                else:
                    await websocket.send_json({"status": agent_manager.status.value})
    except WebSocketDisconnect:
        _ws_clients.discard(websocket)
        logger.info(f"WS client disconnected ({len(_ws_clients)} remaining)")


# ── REST endpoints ──────────────────────────────────────────────────────────

@router.post("/start")
async def start_training(config: TrainingConfig):
    """Start RL agent training."""
    if agent_manager.is_training:
        raise HTTPException(409, "Training already in progress")

    # Get prices for the scenario
    prices = None
    if config.scenario_name:
        price_doc = await firebase.get_prices(config.scenario_name)
        if price_doc and "prices" in price_doc:
            prices = price_doc["prices"]

    # Use mock mode if DWSIM is not available
    use_mock = True  # Set to False when DWSIM is configured

    agent_manager.start_training(
        config=config,
        prices=prices,
        broadcast_fn=_sync_broadcast,
        use_mock=use_mock,
    )

    return {
        "status": "started",
        "config": config.model_dump(),
    }


@router.post("/stop")
async def stop_training():
    """Stop the current training run."""
    agent_manager.stop_training()
    return {"status": "stopped"}


@router.get("/status")
async def training_status():
    """Get current training status with detailed metrics."""
    progress = agent_manager.latest_progress
    # Sanitize numpy types for JSON serialization
    if progress:
        progress = ProgressCallback._to_python(progress)
    return {
        "status": agent_manager.status.value,
        "progress": progress,
        "is_training": agent_manager.is_training,
    }


@router.get("/checkpoints")
async def list_checkpoints():
    """List available model checkpoints with metrics summaries."""
    return agent_manager.list_checkpoints()


@router.post("/load-checkpoint")
async def load_checkpoint(path: str):
    """Load a saved model checkpoint."""
    try:
        agent_manager.load_checkpoint(path)
        return {"status": "loaded", "path": path}
    except FileNotFoundError:
        raise HTTPException(404, f"Checkpoint not found: {path}")


@router.get("/metrics/{run_id}")
async def get_run_metrics(run_id: str):
    """Get full metrics history for a specific training run."""
    data = agent_manager.get_run_metrics(run_id)
    if data is None:
        raise HTTPException(404, f"Metrics not found for run: {run_id}")
    return data


@router.get("/metrics")
async def get_latest_metrics():
    """Get the metrics history for the latest/current training run."""
    data = agent_manager.get_run_metrics()
    if data is None:
        # Return current in-progress metrics if available
        if agent_manager._latest_metrics_history:
            return {
                "run_id": agent_manager._latest_run_id,
                "metrics_history": agent_manager._latest_metrics_history,
            }
        return {"run_id": None, "metrics_history": []}
    return data


@router.post("/optimize", response_model=dict)
async def optimize(request: OptimizationRequest):
    """
    Run the trained agent on current conditions to get optimal action.
    """
    if agent_manager.model is None:
        raise HTTPException(400, "No trained model available")

    # Create environment for observation
    prices = None
    if request.prices:
        prices = request.prices.model_dump(exclude={"timestamp", "scenario_name"})
    elif request.scenario_name:
        price_doc = await firebase.get_prices(request.scenario_name)
        if price_doc and "prices" in price_doc:
            prices = price_doc["prices"]

    env = CDUEnvironment(prices=prices, use_mock=True, max_steps=1)
    obs, info = env.reset()

    # Apply disturbance if specified
    if request.disturbance and not env.use_mock:
        dist_dict = request.disturbance.model_dump()
        env.bridge.apply_disturbance(dist_dict)
        obs = env._get_observation()

    # Get agent's recommendation
    action = agent_manager.predict(obs, deterministic=True)
    real_action = env._denormalize_action(action)
    action_dict = dict(zip(ACTION_KEYS, real_action.tolist()))

    # Simulate the result
    obs_next, reward, _, _, step_info = env.step(action)
    state = step_info.get("state", {})

    # Calculate per-product revenue
    product_revenues = {}
    total_revenue = 0.0
    for prod in PRODUCT_KEYS:
        flow = state.get(f"flow_{prod}", 0.0)
        price = (prices or {}).get(prod, DEFAULT_PRICES.get(prod, 0.0))
        rev = flow * price
        product_revenues[prod] = round(rev, 2)
        total_revenue += rev

    # Calculate feed cost
    feed_flow = state.get("feed_flow_rate", 0.0)
    feed_price = (prices or {}).get("Feed_Crude", DEFAULT_PRICES.get("Feed_Crude", 0.35))
    feed_cost = feed_flow * feed_price

    estimated_profit = total_revenue - feed_cost

    # Gather D95% data from state
    d95_data = {k.replace("d95_", ""): v for k, v in state.items() if k.startswith("d95_")}

    env.close()

    # Save result
    result_data = {
        "action": action_dict,
        "state": state,
        "profit": estimated_profit,
        "product_revenues": product_revenues,
    }
    await firebase.save_optimization_result(result_data)

    return {
        "recommended_action": action_dict,
        "predicted_state": state,
        "estimated_profit": estimated_profit,
        "product_revenues": product_revenues,
        "feed_cost": feed_cost,
        "total_revenue": total_revenue,
        "d95": d95_data,
    }


@router.get("/history")
async def training_history():
    """Get training run history with metrics summaries."""
    return await firebase.get_training_history()
