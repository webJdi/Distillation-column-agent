"""API routes for product price management."""
from fastapi import APIRouter, HTTPException
from backend.models.schemas import ProductPrices, PriceScenario
from backend.services.firebase_service import FirebaseService

router = APIRouter(prefix="/api/prices", tags=["Prices"])
firebase = FirebaseService()


@router.get("/scenarios")
async def list_scenarios():
    """List all saved price scenarios."""
    return await firebase.list_scenarios()


@router.post("/", response_model=dict)
async def save_prices(prices: ProductPrices):
    """Save product prices for a scenario."""
    scenario = prices.scenario_name or "default"
    price_dict = prices.model_dump(exclude={"timestamp", "scenario_name"})
    doc_id = await firebase.save_prices(scenario, price_dict)
    return {"status": "ok", "doc_id": doc_id, "scenario": scenario}


@router.get("/{scenario_name}")
async def get_prices(scenario_name: str = "default"):
    """Get prices for a given scenario."""
    result = await firebase.get_prices(scenario_name)
    if not result:
        raise HTTPException(404, f"Scenario '{scenario_name}' not found")
    return result


@router.post("/scenario", response_model=dict)
async def create_scenario(scenario: PriceScenario):
    """Create a named price scenario."""
    price_dict = scenario.prices.model_dump(exclude={"timestamp", "scenario_name"})
    doc_id = await firebase.save_prices(scenario.name, price_dict)
    return {
        "status": "ok",
        "doc_id": doc_id,
        "name": scenario.name,
        "description": scenario.description,
    }
