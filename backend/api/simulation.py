"""API routes for DWSIM simulation interaction."""
from fastapi import APIRouter, HTTPException
from backend.models.schemas import ColumnState, ColumnAction, FeedDisturbance
from backend.core.dwsim_bridge import DWSIMBridge
from backend.config import settings
from loguru import logger

router = APIRouter(prefix="/api/simulation", tags=["Simulation"])

# Singleton DWSIM bridge — loaded lazily
_bridge: DWSIMBridge | None = None


def get_bridge() -> DWSIMBridge:
    global _bridge
    if _bridge is None:
        _bridge = DWSIMBridge(settings.FLOWSHEET_PATH)
        _bridge.load()
    return _bridge


@router.post("/load")
async def load_flowsheet():
    """Load (or reload) the CDU flowsheet."""
    try:
        bridge = get_bridge()
        bridge.load()
        return {"status": "ok", "message": "Flowsheet loaded"}
    except Exception as exc:
        raise HTTPException(500, f"Failed to load flowsheet: {exc}")


@router.get("/state", response_model=dict)
async def get_state():
    """Get the current column state (all observable variables)."""
    try:
        bridge = get_bridge()
        state = bridge.get_column_state()
        return {"status": "ok", "state": state}
    except Exception as exc:
        raise HTTPException(500, f"Failed to read state: {exc}")


@router.post("/solve")
async def solve():
    """Run the DWSIM solver on the current state."""
    try:
        bridge = get_bridge()
        errors = bridge.solve()
        return {
            "status": "ok" if not errors else "error",
            "errors": errors,
        }
    except Exception as exc:
        raise HTTPException(500, f"Solver failed: {exc}")


@router.post("/apply-action")
async def apply_action(action: ColumnAction):
    """Apply a set of column actions and solve."""
    try:
        bridge = get_bridge()
        action_dict = action.model_dump()
        bridge.apply_action(action_dict)
        errors = bridge.solve()
        state = bridge.get_column_state()
        return {
            "status": "ok" if not errors else "error",
            "errors": errors,
            "state": state,
        }
    except Exception as exc:
        raise HTTPException(500, f"Action application failed: {exc}")


@router.post("/apply-disturbance")
async def apply_disturbance(disturbance: FeedDisturbance):
    """Apply a feed disturbance and solve."""
    try:
        bridge = get_bridge()
        dist_dict = disturbance.model_dump()
        bridge.apply_disturbance(dist_dict)
        errors = bridge.solve()
        state = bridge.get_column_state()
        return {
            "status": "ok" if not errors else "error",
            "errors": errors,
            "state": state,
            "disturbance_applied": dist_dict,
        }
    except Exception as exc:
        raise HTTPException(500, f"Disturbance application failed: {exc}")


@router.get("/products")
async def get_product_flows():
    """Get current product flow rates."""
    try:
        bridge = get_bridge()
        flows = bridge.get_product_flows()
        temps = bridge.get_product_temperatures()
        return {"status": "ok", "flows": flows, "temperatures": temps}
    except Exception as exc:
        raise HTTPException(500, f"Failed to read products: {exc}")


@router.get("/d95")
async def get_d95():
    """Get estimated D95% distillation temperatures for all products."""
    try:
        bridge = get_bridge()
        d95 = bridge.get_d95_all_products()
        return {"status": "ok", "d95": d95}
    except Exception as exc:
        raise HTTPException(500, f"Failed to compute D95%%: {exc}")
