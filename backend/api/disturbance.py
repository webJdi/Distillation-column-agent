"""API routes for feed disturbance management."""
from fastapi import APIRouter, HTTPException
from backend.models.schemas import FeedDisturbance, OptimizationRequest
from backend.core.rl_environment import CDUEnvironment, ACTION_KEYS, PRODUCT_KEYS
from backend.core.rl_agent import RLAgentManager
from backend.services.firebase_service import FirebaseService
from loguru import logger
import numpy as np

router = APIRouter(prefix="/api/disturbance", tags=["Disturbance"])
firebase = FirebaseService()


@router.post("/analyze")
async def analyze_disturbance(disturbance: FeedDisturbance):
    """
    Analyze the impact of a feed disturbance on the CDU.
    If a trained agent is available, shows both the
    uncontrolled impact and the agent's corrective response.
    """
    # Get current prices
    prices = None
    price_doc = await firebase.get_prices("default")
    if price_doc and "prices" in price_doc:
        prices = price_doc["prices"]

    # 1. Baseline (no disturbance)
    env_base = CDUEnvironment(prices=prices, use_mock=True, max_steps=1)
    obs_base, _ = env_base.reset()
    state_base = env_base._raw_state()

    # 2. Disturbed (no agent correction)
    env_dist = CDUEnvironment(prices=prices, use_mock=True, max_steps=1)
    env_dist.reset()

    # Apply disturbance to mock state by modifying curriculum level
    dist_dict = disturbance.model_dump()
    if not env_dist.use_mock:
        env_dist.bridge.apply_disturbance(dist_dict)
    state_dist = env_dist._raw_state()

    # Simulate disturbance effect in mock mode
    if env_dist.use_mock:
        for key, delta_key in [
            ("feed_temperature", "feed_temperature_delta"),
            ("top_temperature", "feed_temperature_delta"),
            ("bottom_temperature", "feed_temperature_delta"),
        ]:
            if delta_key in dist_dict and dist_dict[delta_key] != 0:
                state_dist[key] = state_dist.get(key, 0) + dist_dict[delta_key] * 0.5

        if dist_dict.get("feed_flow_delta", 0) != 0:
            pct = dist_dict["feed_flow_delta"] / 100
            for prod in PRODUCT_KEYS:
                state_dist[f"flow_{prod}"] = state_dist.get(f"flow_{prod}", 0) * (1 + pct * 0.8)
            state_dist["feed_flow_rate"] = state_dist.get("feed_flow_rate", 500) * (1 + pct)

    # 3. Calculate impact
    impact = {}
    for prod in PRODUCT_KEYS:
        base_flow = state_base.get(f"flow_{prod}", 0)
        dist_flow = state_dist.get(f"flow_{prod}", 0)
        change_pct = ((dist_flow - base_flow) / base_flow * 100) if base_flow > 0 else 0
        impact[prod] = {
            "baseline_flow": round(base_flow, 2),
            "disturbed_flow": round(dist_flow, 2),
            "change_percent": round(change_pct, 2),
        }

    # 4. Agent's corrective action (if model available)
    agent_action = None
    agent_state = None
    from backend.api.training import agent_manager
    if agent_manager.model is not None:
        try:
            obs_dist = env_dist._get_observation()
            action = agent_manager.predict(obs_dist, deterministic=True)
            real_action = env_dist._denormalize_action(action)
            agent_action = dict(zip(ACTION_KEYS, real_action.tolist()))
            # The agent would then apply this action — simulate the result
            obs_corrected, reward, _, _, info = env_dist.step(action)
            agent_state = info.get("state", {})
        except Exception as exc:
            logger.warning(f"Agent prediction failed: {exc}")

    # 5. Revenue impact
    base_revenue = sum(
        state_base.get(f"flow_{p}", 0) * (prices or {}).get(p, 0)
        for p in PRODUCT_KEYS
    )
    dist_revenue = sum(
        state_dist.get(f"flow_{p}", 0) * (prices or {}).get(p, 0)
        for p in PRODUCT_KEYS
    )

    env_base.close()
    env_dist.close()

    return {
        "disturbance": dist_dict,
        "product_impact": impact,
        "revenue_impact": {
            "baseline": round(base_revenue, 2),
            "disturbed": round(dist_revenue, 2),
            "change": round(dist_revenue - base_revenue, 2),
            "change_percent": round((dist_revenue - base_revenue) / base_revenue * 100, 2) if base_revenue > 0 else 0,
        },
        "agent_corrective_action": agent_action,
        "agent_corrected_state": agent_state,
    }


@router.get("/presets")
async def disturbance_presets():
    """Return common disturbance presets for quick testing."""
    return [
        {
            "name": "Hot Feed",
            "description": "Feed temperature +20°C (furnace overshoot)",
            "disturbance": {"feed_temperature_delta": 20, "feed_pressure_delta": 0, "feed_flow_delta": 0, "feed_api_gravity_delta": 0},
        },
        {
            "name": "Cold Feed",
            "description": "Feed temperature −15°C (furnace underperformance)",
            "disturbance": {"feed_temperature_delta": -15, "feed_pressure_delta": 0, "feed_flow_delta": 0, "feed_api_gravity_delta": 0},
        },
        {
            "name": "High Throughput",
            "description": "Feed flow +20% (increased crude processing)",
            "disturbance": {"feed_temperature_delta": 0, "feed_pressure_delta": 0, "feed_flow_delta": 20, "feed_api_gravity_delta": 0},
        },
        {
            "name": "Low Throughput",
            "description": "Feed flow −15% (reduced processing)",
            "disturbance": {"feed_temperature_delta": 0, "feed_pressure_delta": 0, "feed_flow_delta": -15, "feed_api_gravity_delta": 0},
        },
        {
            "name": "Light Crude Switch",
            "description": "API gravity +5 (switching to lighter crude)",
            "disturbance": {"feed_temperature_delta": 0, "feed_pressure_delta": 0, "feed_flow_delta": 0, "feed_api_gravity_delta": 5},
        },
        {
            "name": "Heavy Crude Switch",
            "description": "API gravity −5 (switching to heavier crude)",
            "disturbance": {"feed_temperature_delta": 0, "feed_pressure_delta": 0, "feed_flow_delta": 0, "feed_api_gravity_delta": -5},
        },
        {
            "name": "Pressure Surge",
            "description": "Feed pressure +30 kPa",
            "disturbance": {"feed_temperature_delta": 0, "feed_pressure_delta": 30, "feed_flow_delta": 0, "feed_api_gravity_delta": 0},
        },
        {
            "name": "Combined Harsh",
            "description": "Hot feed + heavy crude + high throughput",
            "disturbance": {"feed_temperature_delta": 25, "feed_pressure_delta": 10, "feed_flow_delta": 15, "feed_api_gravity_delta": -4},
        },
    ]
