"""API routes for the AI Agent — explanations, reports, Q&A."""
from fastapi import APIRouter, HTTPException
from backend.models.schemas import AIQuery, AIResponse, ReportRequest, ReportResponse
from backend.core.ai_agent import AIAgent
from backend.services.firebase_service import FirebaseService
from datetime import datetime
from loguru import logger

router = APIRouter(prefix="/api/ai", tags=["AI Agent"])

ai_agent = AIAgent()
firebase = FirebaseService()


@router.post("/ask", response_model=dict)
async def ask_ai(query: AIQuery):
    """Ask the AI agent a question about the CDU system."""
    # Build context
    context = {}
    if query.include_current_state:
        # Get latest prices
        price_doc = await firebase.get_prices("default")
        if price_doc and "prices" in price_doc:
            context["prices"] = price_doc["prices"]

        # Get latest training progress
        from backend.api.training import agent_manager
        if agent_manager.latest_progress:
            context["training_progress"] = agent_manager.latest_progress

    if query.context:
        try:
            import json
            extra = json.loads(query.context) if isinstance(query.context, str) else query.context
            context.update(extra)
        except Exception:
            pass

    result = await ai_agent.ask(query.question, context=context)
    return result


@router.post("/report", response_model=dict)
async def generate_report(request: ReportRequest):
    """Generate a structured report."""
    # Gather data for the report
    data = {}

    for scenario in request.scenario_names:
        price_doc = await firebase.get_prices(scenario)
        if price_doc:
            data[f"prices_{scenario}"] = price_doc

    # Get optimization history
    opt_history = await firebase.get_optimization_history(limit=10)
    if opt_history:
        data["optimization_history"] = opt_history

    # Get training history
    train_history = await firebase.get_training_history(limit=5)
    if train_history:
        data["training_history"] = train_history

    result = await ai_agent.generate_report(
        report_type=request.report_type,
        data=data,
    )
    return result


@router.post("/clear-history")
async def clear_conversation():
    """Clear the AI agent's conversation history."""
    ai_agent.clear_history()
    return {"status": "ok", "message": "Conversation history cleared"}


@router.get("/capabilities")
async def agent_capabilities():
    """List what the AI agent can do."""
    return {
        "capabilities": [
            {
                "name": "Q&A",
                "description": "Ask questions about the CDU, distillation theory, RL training, optimization results",
                "endpoint": "/api/ai/ask",
            },
            {
                "name": "Reports",
                "description": "Generate summary, detailed, optimization, or comparison reports",
                "endpoint": "/api/ai/report",
                "types": ["summary", "detailed", "optimization", "comparison"],
            },
            {
                "name": "Explain Actions",
                "description": "Understand why the RL agent recommends specific column settings",
                "endpoint": "/api/ai/ask",
                "example_query": "Why did the agent increase the reflux ratio?",
            },
            {
                "name": "Safety Analysis",
                "description": "Get safety assessments of current or proposed operating conditions",
                "endpoint": "/api/ai/ask",
                "example_query": "Is it safe to increase the bottom temperature to 380°C?",
            },
        ],
        "offline_mode": ai_agent.client is None,
    }
