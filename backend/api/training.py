from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Optional
import asyncio
import glob
import json
import os
import queue
import threading

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
from backend.core.kpi_accumulator import KPIAccumulator
from backend.services.firebase_service import FirebaseService
from loguru import logger
import numpy as np

router = APIRouter(prefix="/api/training", tags=["Training"])

# Singletons
agent_manager = RLAgentManager()
firebase = FirebaseService()

# Connected WebSocket clients for live progress
_ws_clients: set[WebSocket] = set()
_broadcast_queue: queue.Queue = queue.Queue()


async def _broadcast_to_ws(data: dict):
    """Send training progress to all connected WebSocket clients."""
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


def _sync_broadcast(data: dict):
    """
    Thread-safe broadcast: puts message on queue to be consumed by async loop.
    Called from training thread.
    """
    try:
        _broadcast_queue.put_nowait(data)
    except queue.Full:
        pass  # Queue is full, drop this message


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

    # Notebook-equivalent training path uses real DWSIM by default.
    use_mock = False

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
    """Load a saved model checkpoint, stopping any active training first."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, agent_manager.load_checkpoint, path)
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
    Run the trained agent on current conditions to get optimal action(s).
    If num_runs > 1, runs multiple times and returns average ± std dev,
    with recommended settings from the best (highest profit) run.
    """
    if agent_manager.model is None:
        raise HTTPException(400, "No trained model available")

    # Parse num_runs (default 1)
    num_runs = max(1, min(request.num_runs or 1, 200))  # Cap at 200
    use_stochastic = num_runs > 1

    # Create environment for observation
    prices = None
    if request.prices:
        prices = request.prices.model_dump(exclude={"timestamp", "scenario_name"})
    elif request.scenario_name:
        price_doc = await firebase.get_prices(request.scenario_name)
        if price_doc and "prices" in price_doc:
            prices = price_doc["prices"]

    # Run multiple times and collect results
    all_profits = []
    all_revenues = []
    all_feed_costs = []
    all_product_revenues = []
    all_states = []
    all_actions = []
    best_profit = float("-inf")
    best_run_idx = 0

    for run_idx in range(num_runs):
        env = CDUEnvironment(prices=prices, use_mock=True, max_steps=1)
        obs, info = env.reset()

        # Apply disturbance if specified
        if request.disturbance and not env.use_mock:
            dist_dict = request.disturbance.model_dump()
            env.bridge.apply_disturbance(dist_dict)
            obs = env._get_observation()

        # Get agent's recommendation (deterministic=False for stochastic results when num_runs > 1)
        action = agent_manager.predict(obs, deterministic=not use_stochastic)
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

        profit = total_revenue - feed_cost

        # Collect results
        all_profits.append(profit)
        all_revenues.append(total_revenue)
        all_feed_costs.append(feed_cost)
        all_product_revenues.append(product_revenues)
        all_states.append(state)
        all_actions.append(action_dict)

        # Track best run
        if profit > best_profit:
            best_profit = profit
            best_run_idx = run_idx

        env.close()

    # Calculate statistics
    avg_profit = float(np.mean(all_profits))
    std_profit = float(np.std(all_profits)) if num_runs > 1 else 0.0
    avg_revenue = float(np.mean(all_revenues))
    avg_feed_cost = float(np.mean(all_feed_costs))

    # Use best run's state and action for recommendations
    best_action_dict = all_actions[best_run_idx]
    best_state = all_states[best_run_idx]

    # Average product revenues across all runs
    avg_product_revenues = {}
    for prod in PRODUCT_KEYS:
        revenues = [pr.get(prod, 0.0) for pr in all_product_revenues]
        avg_product_revenues[prod] = round(float(np.mean(revenues)), 2)

    # Gather D95% data from best run's state
    d95_data = {k.replace("d95_", ""): v for k, v in best_state.items() if k.startswith("d95_")}

    # Save best result
    result_data = {
        "action": best_action_dict,
        "state": best_state,
        "profit": best_profit,
        "product_revenues": all_product_revenues[best_run_idx],
        "num_runs": num_runs,
        "avg_profit": avg_profit,
        "std_profit": std_profit,
    }
    await firebase.save_optimization_result(result_data)

    # Record KPI data once (using best run)
    try:
        baseline_env = CDUEnvironment(prices=prices, use_mock=True, max_steps=1)
        baseline_obs, _ = baseline_env.reset()
        baseline_state = baseline_env._raw_state()
        baseline_profit = sum(
            baseline_state.get(f"flow_{p}", 0.0) * (prices or {}).get(p, DEFAULT_PRICES.get(p, 0.0))
            for p in PRODUCT_KEYS
        ) - baseline_state.get("feed_flow_rate", 0.0) * (prices or {}).get("Feed_Crude", DEFAULT_PRICES.get("Feed_Crude", 0.35))
        
        KPIAccumulator.record_run(
            base_state=baseline_state,
            opt_state=best_state,
            base_profit=baseline_profit,
            opt_profit=best_profit,
            reboiler_duty_base=baseline_state.get("reboiler_duty", 0.0),
            reboiler_duty_opt=best_state.get("reboiler_duty", 0.0),
        )
        baseline_env.close()
    except Exception as exc:
        logger.warning(f"KPI recording failed: {exc}")

    return {
        "recommended_action": best_action_dict,
        "predicted_state": best_state,
        "estimated_profit": best_profit,  # For single run
        "avg_profit": avg_profit,  # Average across runs
        "std_profit": std_profit,  # Std dev across runs
        "best_profit": best_profit,  # Best (highest) profit
        "avg_revenue": avg_revenue,
        "avg_feed_cost": avg_feed_cost,
        "product_revenues": avg_product_revenues,
        "total_revenue": avg_revenue,
        "feed_cost": avg_feed_cost,
        "d95": d95_data,
        "num_runs": num_runs,
    }


@router.get("/optimization-scope")
async def optimization_scope():
    """
    Compare the current operating state with the best action the RL agent can find.

    Strategy — "look harder, or stay put":
      1. Draw N stochastic action samples and one deterministic sample.
      2. Keep the sample that achieves the highest profit.
      3. If even the best sample does not beat the baseline, the agent
         recommends no change (delta = 0).  The scope card shows a flat line
         rather than a misleading negative improvement.
      4. KPI is only recorded when a genuine positive improvement is found.
    """
    if agent_manager.model is None:
        return {"available": False, "reason": "No trained model loaded"}

    try:
        price_doc = await firebase.get_prices("default")
        prices = price_doc.get("prices") if price_doc else None

        env = CDUEnvironment(prices=prices, use_mock=True, max_steps=1)
        obs, _ = env.reset()
        state_base = env._raw_state()

        def _rev(state):
            r = sum(
                state.get(f"flow_{p}", 0.0)
                * (prices or {}).get(p, DEFAULT_PRICES.get(p, 0.0))
                for p in PRODUCT_KEYS
            )
            feed = state.get("feed_flow_rate", 0.0) * (
                (prices or {}).get("Feed_Crude", DEFAULT_PRICES.get("Feed_Crude", 0.35))
            )
            return r, feed

        base_rev, feed_cost = _rev(state_base)
        base_profit = base_rev - feed_cost

        # --- Look harder: try deterministic + 4 stochastic samples, keep best ---
        _SAMPLES = 5
        best_profit = float("-inf")
        best_state_opt = None
        best_action_dict = None

        for i in range(_SAMPLES):
            deterministic = (i == 0)   # first pass is deterministic, rest stochastic
            action = agent_manager.predict(obs, deterministic=deterministic)
            real_action = env._denormalize_action(action)
            action_dict_candidate = dict(zip(ACTION_KEYS, real_action.tolist()))
            _, _, _, _, step_info = env.step(action)
            state_candidate = step_info.get("state", {})
            cand_rev, _ = _rev(state_candidate)
            cand_profit = cand_rev - feed_cost
            if cand_profit > best_profit:
                best_profit = cand_profit
                best_state_opt = state_candidate
                best_action_dict = action_dict_candidate
            # reset env to baseline before next sample
            env.reset()

        env.close()

        delta_profit = best_profit - base_profit

        # --- Stay put if no genuine improvement found ---
        if delta_profit <= 0:
            prog = agent_manager.latest_progress or {}
            return {
                "available": True,
                "baseline_profit": round(base_profit, 2),
                "optimized_profit": round(base_profit, 2),
                "delta_profit": 0.0,
                "delta_pct": 0.0,
                "product_delta": {p: 0.0 for p in PRODUCT_KEYS},
                "recommendations": {},
                "no_improvement": True,
                "model_info": {
                    "algorithm": prog.get("config", {}).get("algorithm", "SAC"),
                    "best_reward": round(prog.get("best_reward", 0.0), 4) if prog else None,
                },
            }

        delta_pct = (delta_profit / abs(base_profit) * 100) if base_profit != 0 else 0.0

        product_delta = {
            p: round(
                best_state_opt.get(f"flow_{p}", 0.0)
                * (prices or {}).get(p, DEFAULT_PRICES.get(p, 0.0))
                - state_base.get(f"flow_{p}", 0.0)
                * (prices or {}).get(p, DEFAULT_PRICES.get(p, 0.0)),
                2,
            )
            for p in PRODUCT_KEYS
        }

        # Only record KPI when a genuine improvement exists
        KPIAccumulator.record_run(
            base_state=state_base,
            opt_state=best_state_opt,
            base_profit=base_profit,
            opt_profit=best_profit,
            reboiler_duty_base=state_base.get("reboiler_duty", 0.0) + state_base.get("vac_reboiler_duty", 0.0),
            reboiler_duty_opt=best_state_opt.get("reboiler_duty", 0.0) + best_state_opt.get("vac_reboiler_duty", 0.0),
            furnace_duty_base=state_base.get("furnace_duty", 0.0),
            furnace_duty_opt=best_state_opt.get("furnace_duty", 0.0),
        )

        prog = agent_manager.latest_progress or {}
        return {
            "available": True,
            "baseline_profit": round(base_profit, 2),
            "optimized_profit": round(best_profit, 2),
            "delta_profit": round(delta_profit, 2),
            "delta_pct": round(delta_pct, 2),
            "product_delta": product_delta,
            "recommendations": {k: round(v, 3) for k, v in best_action_dict.items()},
            "no_improvement": False,
            "model_info": {
                "algorithm": prog.get("config", {}).get("algorithm", "SAC"),
                "best_reward": round(prog.get("best_reward", 0.0), 4) if prog else None,
            },
        }
    except Exception as exc:
        logger.error(f"optimization_scope error: {exc}")
        return {"available": False, "reason": str(exc)}


@router.get("/kpi-stats")
async def get_kpi_stats():
    """
    Get accumulated KPI statistics across all optimization runs.
    Returns: avg_distillate_yield_improvement, avg_energy_savings,
    cumulative_profit, run_count.
    """
    return KPIAccumulator.get_stats()


@router.post("/kpi-reset")
async def reset_kpi_stats():
    """Reset all saved KPI summaries and history."""
    KPIAccumulator.reset()
    return KPIAccumulator.get_stats()


@router.get("/history")
async def training_history():
    """Get training run history with metrics summaries."""
    return await firebase.get_training_history()
