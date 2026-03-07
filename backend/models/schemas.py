"""Pydantic models / schemas for the CDU Optimizer API."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# ── Product prices ──────────────────────────────────────────────────────────

class ProductPrices(BaseModel):
    """Market prices for each CDU+NSU+VDU product in $/kg.

    All prices are in $/kg to be directly compatible with DWSIM mass flow rates
    (kg/h) in the reward calculation: profit = Σ flow(kg/h) × price($/kg) − costs.
    """
    Uncondensed_Gas: float = Field(ge=0, description="Uncondensed Gas price $/kg")
    Heavy_Naphtha: float = Field(ge=0, description="Heavy Naphtha price $/kg")
    SKO: float = Field(ge=0, description="Superior Kerosene Oil (Jet Fuel) price $/kg")
    Light_Gas_Oil: float = Field(ge=0, description="Light Gas Oil / Light Diesel price $/kg")
    Heavy_Gas_Oil: float = Field(ge=0, description="Heavy Gas Oil / Heavy Diesel price $/kg")
    StabOffGas: float = Field(ge=0, description="Stabilizer Off-Gas price $/kg")
    LPG: float = Field(ge=0, description="LPG price $/kg")
    SRN: float = Field(ge=0, description="Straight-Run Naphtha price $/kg")
    Offgas: float = Field(ge=0, description="VDU Overhead Off-Gas price $/kg")
    Vacuum_Diesel: float = Field(ge=0, description="Vacuum Diesel price $/kg")
    Vacuum_Gas_Oil: float = Field(ge=0, description="Vacuum Gas Oil price $/kg")
    Hotwell_Oil: float = Field(ge=0, description="Hotwell Oil / Slop Cut price $/kg")
    Vac_residue: float = Field(ge=0, description="Vacuum Residue price $/kg")
    Feed_Crude: float = Field(default=0.45, ge=0, description="Crude oil feed cost $/kg")
    timestamp: Optional[datetime] = None
    scenario_name: Optional[str] = "default"


class PriceScenario(BaseModel):
    """A saved pricing scenario."""
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    prices: ProductPrices
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ── Simulation state ────────────────────────────────────────────────────────

class ColumnState(BaseModel):
    """Observable state of the distillation columns (ADU + NSU + VDU)."""
    # Product flow rates (kg/h) — 13 products
    flow_Uncondensed_Gas: float = 0.0
    flow_Heavy_Naphtha: float = 0.0
    flow_SKO: float = 0.0
    flow_Light_Gas_Oil: float = 0.0
    flow_Heavy_Gas_Oil: float = 0.0
    flow_StabOffGas: float = 0.0
    flow_LPG: float = 0.0
    flow_SRN: float = 0.0
    flow_Offgas: float = 0.0
    flow_Vacuum_Diesel: float = 0.0
    flow_Vacuum_Gas_Oil: float = 0.0
    flow_Hotwell_Oil: float = 0.0
    flow_Vac_residue: float = 0.0

    # Key temperatures (°C)
    top_temperature: float = 0.0
    bottom_temperature: float = 0.0
    feed_temperature: float = 0.0

    # Key pressures (kPa)
    top_pressure: float = 0.0
    bottom_pressure: float = 0.0

    # Atmospheric column performance
    condenser_duty: float = 0.0  # kW
    reboiler_duty: float = 0.0   # kW

    # Vacuum column state
    vac_top_pressure: float = 0.0
    vac_bottom_pressure: float = 0.0
    vac_condenser_duty: float = 0.0
    vac_reboiler_duty: float = 0.0
    vac_bottom_temperature: float = 0.0

    # Feed properties
    feed_flow_rate: float = 0.0  # kg/h
    feed_api_gravity: float = 0.0


class ColumnAction(BaseModel):
    """Actions the RL agent can take on all three columns."""
    # --- Atmospheric column ---
    reflux_ratio: float = Field(ge=0.3, le=20.0, description="ADU reflux ratio (Specs C)")
    hn_draw_temp: float = Field(ge=100, le=300, description="Heavy Naphtha draw temperature (°C)")
    sko_draw_temp: float = Field(ge=120, le=310, description="SKO draw temperature (°C)")
    ld_draw_temp: float = Field(ge=150, le=350, description="Light Gas Oil draw temperature (°C)")
    hd_draw_temp: float = Field(ge=200, le=380, description="Heavy Gas Oil draw temperature (°C)")
    atmos_reboiler_temp: float = Field(ge=250, le=450, description="Atmospheric reboiler temperature spec (°C)")
    # --- Naphtha Stabilizer ---
    nsu_reflux_ratio: float = Field(ge=0.2, le=20.0, description="NSU reflux ratio (Specs C)")
    nsu_reboiler_temp: float = Field(ge=100, le=250, description="NSU reboiler temperature spec (°C)")
    # --- Vacuum column ---
    vac_reflux_ratio: float = Field(ge=0.2, le=20.0, description="VDU reflux ratio (Specs C)")
    vac_reboiler_temp: float = Field(ge=250, le=450, description="Vacuum reboiler temperature spec (°C)")
    vac_diesel_draw_temp: float = Field(ge=100, le=350, description="Vacuum Diesel draw temperature (°C)")
    vgo_draw_temp: float = Field(ge=150, le=400, description="VGO draw temperature (°C)")


# ── Disturbances ────────────────────────────────────────────────────────────

class FeedDisturbance(BaseModel):
    """Disturbance parameters the user can introduce."""
    feed_temperature_delta: float = Field(
        default=0.0, ge=-50, le=50,
        description="Change in feed temperature (°C)"
    )
    feed_pressure_delta: float = Field(
        default=0.0, ge=-50, le=50,
        description="Change in feed pressure (kPa)"
    )
    feed_flow_delta: float = Field(
        default=0.0, ge=-30, le=30,
        description="Change in feed flow rate (%)"
    )
    feed_api_gravity_delta: float = Field(
        default=0.0, ge=-10, le=10,
        description="Change in feed API gravity"
    )
    crude_blend: Optional[str] = Field(
        default=None,
        description="Crude oil type (e.g., 'WTI_Light', 'Azeri Light')"
    )


# ── RL Training ─────────────────────────────────────────────────────────────

class TrainingStatus(str, Enum):
    IDLE = "idle"
    TRAINING = "training"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class TrainingConfig(BaseModel):
    """Configuration for RL training run."""
    algorithm: str = Field(default="SAC", description="RL algorithm (SAC, PPO, TD3)")
    total_timesteps: int = Field(default=50_000, ge=1000, le=1_000_000)
    learning_rate: float = Field(default=3e-4, ge=1e-6, le=1e-1)
    batch_size: int = Field(default=256, ge=32, le=2048)
    gamma: float = Field(default=0.99, ge=0.9, le=0.999)
    scenario_name: Optional[str] = "default"
    use_curriculum: bool = Field(default=True, description="Use curriculum learning")


class TrainingProgress(BaseModel):
    """Real-time training progress update with detailed metrics."""
    status: TrainingStatus = TrainingStatus.IDLE
    current_step: int = 0
    total_steps: int = 0
    episode: int = 0
    episode_reward: float = 0.0
    avg_reward: float = 0.0
    best_reward: float = float("-inf")
    profit: float = 0.0

    # SAC-specific losses
    critic_loss: Optional[float] = None
    actor_loss: Optional[float] = None
    ent_coef: Optional[float] = None       # alpha
    ent_coef_loss: Optional[float] = None
    n_updates: Optional[int] = None

    # Derived metrics
    mean_q_value: Optional[float] = None
    entropy: Optional[float] = None

    # Gradient norms
    actor_grad_norm: Optional[float] = None
    critic_grad_norm: Optional[float] = None

    # Replay buffer
    replay_buffer_size: Optional[int] = None
    replay_buffer_capacity: Optional[int] = None
    replay_buffer_pct: Optional[float] = None

    # Action distribution
    action_distribution: Optional[dict] = None

    # Run metadata
    run_id: Optional[str] = None
    training_time_seconds: Optional[float] = None
    checkpoint_path: Optional[str] = None
    checkpoint_size_mb: Optional[float] = None
    config: Optional[dict] = None

    losses: dict = {}
    timestamp: Optional[datetime] = None


class TrainingResult(BaseModel):
    """Final training result."""
    checkpoint_path: str
    total_episodes: int
    best_reward: float
    avg_reward_last_100: float
    final_profit: float
    training_time_seconds: float
    config: TrainingConfig


# ── Optimization ────────────────────────────────────────────────────────────

class OptimizationRequest(BaseModel):
    """Request to run the trained agent on a scenario."""
    scenario_name: Optional[str] = "default"
    disturbance: Optional[FeedDisturbance] = None
    prices: Optional[ProductPrices] = None


class OptimizationResult(BaseModel):
    """Result of running the trained agent."""
    recommended_action: ColumnAction
    predicted_state: ColumnState
    estimated_profit: float
    product_revenues: dict[str, float]
    energy_cost: float
    confidence: float = Field(ge=0, le=1)
    explanation: Optional[str] = None


# ── AI Agent ────────────────────────────────────────────────────────────────

class AIQuery(BaseModel):
    """Query for the AI agent."""
    question: str
    context: Optional[str] = None
    include_current_state: bool = True


class AIResponse(BaseModel):
    """Response from the AI agent."""
    answer: str
    sources: list[str] = []
    suggested_actions: list[str] = []


class ReportRequest(BaseModel):
    """Request to generate a system report."""
    report_type: str = Field(
        default="summary",
        description="Type: summary, detailed, optimization, comparison"
    )
    scenario_names: list[str] = ["default"]
    include_charts: bool = True


class ReportResponse(BaseModel):
    """Generated report metadata."""
    report_id: str
    report_type: str
    file_path: str
    created_at: datetime
    summary: str
