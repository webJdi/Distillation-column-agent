"""
Gymnasium environment for the Crude Distillation Unit (CDU + NSU + VDU).

The main_sim.dwxmz flowsheet has three columns:
  1. Atmospheric Distillation Column — produces Uncondensed_Gas,
     Heavy_Naphtha, SKO, Light_Gas_Oil, Heavy_Gas_Oil
  2. Naphtha Stabilizer — produces StabOffGas, LPG, SRN
  3. Vacuum Distillation Column — produces Offgas, Vacuum_Diesel,
     Vacuum_Gas_Oil, Hotwell_Oil, Vac_residue

Observation space (37-dim continuous):
    13 product flow rates + 13 product temperatures +
    feed_temperature + feed_flow + top_pressure + bottom_pressure +
    top_temperature + bottom_temperature + condenser_duty +
    vac_top_pressure + vac_bottom_pressure + vac_condenser_duty +
    vac_bottom_temperature

Action space (12-dim, delta-action paradigm):
    Each action value is a per-step *delta* in physical units.
    The bridge reads the current DWSIM value, adds the delta, and clamps
    to hard limits — keeping the solver in its convergence basin.
    reflux_ratio, hn_draw_temp, sko_draw_temp, ld_draw_temp, hd_draw_temp,
    atmos_reboiler_temp, nsu_reflux_ratio, nsu_reboiler_temp,
    vac_reflux_ratio, vac_reboiler_temp, vac_diesel_draw_temp, vgo_draw_temp

Reward = Σ (product_flow × product_price) − feed_cost − d95_penalty
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional
from loguru import logger

from backend.core.dwsim_bridge import (
    DWSIMBridge, ATMOS_COLUMN_NAME, NSU_COLUMN_NAME, VAC_COLUMN_NAME,
)
from backend.config import settings


# ── Normalization helpers ───────────────────────────────────────────────────
# These are approximate ranges used to normalize observations to [0, 1].
OBS_RANGES = {
    "flow":        (0.0, 70000.0),   # kg/h  (VDU overhead Offgas can reach ~66k kg/h)
    "temp":        (0.0, 500.0),     # °C
    "pressure":    (50.0, 400.0),    # kPa
    "vac_pressure":(1.0, 50.0),      # kPa (vacuum side)
    "feed_flow":   (0.0, 10000.0),   # kg/h
    "duty":        (0.0, 50_000.0),  # kW
}

# ── Delta-action paradigm ────────────────────────────────────────────────────
# The RL action in [-1, 1] is multiplied by ACTION_SCALES to get a per-step
# *delta* in physical units (°C for temperatures, dimensionless for reflux).
# The bridge reads the current DWSIM value, adds the delta, then hard-clamps.
#
# Per-step deltas are intentionally small so the DWSIM Newton solver stays
# inside its convergence basin at every step.  A solver failure is treated as
# a natural episode termination — the flowsheet is reloaded from the saved
# baseline and a new episode begins with a fresh (different) action sequence.
#
# Rule of thumb: max delta per step ≈ 0.5 % of the nominal operating range.
ACTION_SCALES: dict[str, float] = {
    # --- Atmospheric column ---
    "reflux_ratio":         0.05,   # ±0.05   (base ~5.0)
    "hn_draw_temp":         0.5,    # ±0.5 °C (base ~242 °C)
    "sko_draw_temp":        0.5,    # ±0.5 °C (base ~244 °C)
    "ld_draw_temp":         0.5,    # ±0.5 °C (base ~247 °C)
    "hd_draw_temp":         0.5,    # ±0.5 °C (base ~248 °C)
    "atmos_reboiler_temp":  1.0,    # ±1.0 °C (base ~365 °C)
    "atmos_top_pressure":   0.5,    # ±0.5 kPa (base ~101 kPa)
    "atmos_dp":             0.2,    # ±0.2 kPa (base ~15 kPa)
    # --- Naphtha Stabilizer ---
    "nsu_reflux_ratio":     0.05,   # ±0.05   (base ~5.0)
    "nsu_reboiler_temp":    0.5,    # ±0.5 °C (base ~155 °C)
    # --- Vacuum column ---
    "vac_reflux_ratio":     0.05,   # ±0.05   (base ~5.0)
    "vac_reboiler_temp":    1.0,    # ±1.0 °C (base ~360 °C)
    "vac_diesel_draw_temp": 0.5,    # ±0.5 °C
    "vgo_draw_temp":        0.5,    # ±0.5 °C
    "vac_top_pressure":     0.1,    # ±0.1 kPa (base ~8 kPa)
    "vac_dp":               0.1,    # ±0.1 kPa (base ~7 kPa)
}

# Absolute hard limits — bridge clamps the running value after each delta.
# Prevents multi-step accumulation from drifting outside safe process bounds.
ACTION_HARD_LIMITS: dict[str, tuple[float, float]] = {
    "reflux_ratio":         (1.0,   12.0),
    "hn_draw_temp":         (100.0, 280.0),
    "sko_draw_temp":        (150.0, 310.0),
    "ld_draw_temp":         (180.0, 345.0),
    "hd_draw_temp":         (200.0, 370.0),
    "atmos_reboiler_temp":  (330.0, 415.0),
    "atmos_top_pressure":   (80.0,  130.0),   # kPa
    "atmos_dp":             (5.0,   35.0),    # kPa
    "nsu_reflux_ratio":     (1.0,   12.0),
    "nsu_reboiler_temp":    (120.0, 200.0),
    "vac_reflux_ratio":     (1.0,   12.0),
    "vac_reboiler_temp":    (330.0, 415.0),
    "vac_diesel_draw_temp": (150.0, 310.0),
    "vgo_draw_temp":        (200.0, 380.0),
    "vac_top_pressure":     (2.0,   25.0),    # kPa
    "vac_dp":               (1.0,   15.0),    # kPa
}

# Kept for API compatibility — col 0 = -scale, col 1 = +scale (delta range).
# Populated after ACTION_KEYS is defined below.
FALLBACK_ACTION_RANGES = None
ACTION_RANGES = None

ACTION_KEYS = [
    # Atmospheric column (8)
    "reflux_ratio", "hn_draw_temp",
    "sko_draw_temp", "ld_draw_temp", "hd_draw_temp",
    "atmos_reboiler_temp", "atmos_top_pressure", "atmos_dp",
    # Naphtha Stabilizer (2)
    "nsu_reflux_ratio", "nsu_reboiler_temp",
    # Vacuum column (6)
    "vac_reflux_ratio", "vac_reboiler_temp",
    "vac_diesel_draw_temp", "vgo_draw_temp",
    "vac_top_pressure", "vac_dp",
]

PRODUCT_KEYS = [
    "Uncondensed_Gas", "Heavy_Naphtha", "SKO", "Light_Gas_Oil", "Heavy_Gas_Oil",
    "StabOffGas", "LPG", "SRN",
    "Offgas", "Vacuum_Diesel", "Vacuum_Gas_Oil", "Hotwell_Oil", "Vac_residue",
]

# Default product prices ($/kg) — will be overridden by user prices
DEFAULT_PRICES = {
    "Uncondensed_Gas": 0.30,   # refinery fuel gas
    "Heavy_Naphtha":   0.60,   # reformer feed
    "SKO":             0.75,   # jet fuel / kerosene
    "Light_Gas_Oil":   0.70,   # light diesel
    "Heavy_Gas_Oil":   0.70,   # heavy diesel
    "StabOffGas":      0.30,   # stabilizer off-gas
    "LPG":             0.65,   # liquefied petroleum gas
    "SRN":             0.75,   # straight-run naphtha
    "Offgas":          0.30,   # VDU overhead gas
    "Vacuum_Diesel":   0.70,   # VDU diesel
    "Vacuum_Gas_Oil":  0.50,   # FCC feed
    "Hotwell_Oil":     0.50,   # slop cut
    "Vac_residue":     0.35,   # Bitumen
    "Feed_Crude":      0.40,   # crude oil feed cost $/kg
}

# ── D95% specification limits (°C) ─────────────────────────────────────────
# Products whose D95% exceeds the spec receive a penalty in the reward.
# Gas/light streams and slop cuts excluded (no meaningful D95 spec).
# User can edit these thresholds later.
D95_SPECS: dict[str, float] = {
    "Heavy_Naphtha":   220.0,
    "SKO":             300.0,  # jet fuel spec (ASTM D1655)
    "Light_Gas_Oil":   370.0,  # diesel spec (EN 590)
    "Heavy_Gas_Oil":   385.0,
    "Vacuum_Diesel":   385.0,
    "Vacuum_Gas_Oil":  520.0,
}

# ── Progressive action warmup ──────────────────────────────────────────────────
# Initial steps use very small deltas to let the solver stay converged.
# The scale ramps linearly from WARMUP_INITIAL to 1.0 over the first
# WARMUP_END_FRAC of the episode.  This keeps early exploration safe and
# lets the agent make larger, more decisive moves later when it has a
# better sense of the profit gradient.
WARMUP_INITIAL:  float = 0.05   # 5 % of full scale at step 1
WARMUP_END_FRAC: float = 0.7    # reach 100 % by 70 % of max_steps

# ── Adaptive solver tolerance defaults ──────────────────────────────────
# On solver failure the tolerance is multiplied by TOL_ESCALATION_FACTOR
# and the solve is retried once.  After every successful step the
# tolerance is slowly tightened by TOL_RELAX_FACTOR toward TOL_MIN.
TOL_MIN:               float = 0.1
TOL_MAX:               float = 100.0
TOL_INITIAL:           float = 0.5    # conservative starting point
TOL_ESCALATION_FACTOR: float = 3.0    # multiply on failure
TOL_RELAX_FACTOR:      float = 0.85   # multiply on success (tighten)


class CDUEnvironment(gym.Env):
    """
    Gymnasium environment wrapping a DWSIM CDU + NSU + VDU flowsheet.

    The agent learns to set column operating parameters to maximize
    profit = revenue(product flows × prices) − feed cost − D95% penalties.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        flowsheet_path: Optional[str] = None,
        prices: Optional[dict[str, float]] = None,
        max_steps: int = 200,
        curriculum_level: float = 1.0,   # 0→easy, 1→full difficulty
        use_mock: bool = False,          # True = skip DWSIM (for unit tests)
    ):
        super().__init__()

        self.max_steps = max_steps
        self.current_step = 0
        self.curriculum_level = np.clip(curriculum_level, 0.0, 1.0)
        self.prices = prices or DEFAULT_PRICES.copy()
        self.use_mock = use_mock

        # Observation: 37-dimensional (13 flows + 13 temps + 11 column state)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(37,), dtype=np.float32
        )

        # Action: 16-dimensional, normalized to [-1, 1] (SB3 convention)
        # 8 ADU params + 2 NSU params + 6 VDU params
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )

        # DWSIM bridge
        if not use_mock:
            self.bridge = DWSIMBridge(flowsheet_path)
        else:
            self.bridge = None

        self._operating_point: Optional[dict] = None

        # Episode tracking
        self._episode_reward = 0.0
        self._best_profit = float("-inf")
        self._last_state: Optional[dict] = None
        self._last_action: Optional[dict] = None

        # Adaptive solver tolerance (per-column)
        self._solver_tol: dict[str, float] = {
            ATMOS_COLUMN_NAME: TOL_INITIAL,
            NSU_COLUMN_NAME:   TOL_INITIAL,
            VAC_COLUMN_NAME:   TOL_INITIAL,
        }

        logger.info(
            f"CDUEnvironment created | mock={use_mock} | "
            f"max_steps={max_steps} | curriculum={curriculum_level:.2f}"
        )

    # ── Gym API ─────────────────────────────────────────────────────────

    # Episode counter (survives resets)
    _episode_number: int = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        CDUEnvironment._episode_number += 1
        ep = CDUEnvironment._episode_number
        self.current_step = 0
        self._episode_reward = 0.0

        if not self.use_mock:
            logger.info(
                f"{'='*55}\n"
                f" Episode {ep} START — reloading flowsheet to baseline\n"
                f"{'='*55}"
            )
            self.bridge.load()   # re-reads .dwxmz from disk → clean baseline
            # Reset solver tolerances to initial values
            self._solver_tol = {
                ATMOS_COLUMN_NAME: TOL_INITIAL,
                NSU_COLUMN_NAME:   TOL_INITIAL,
                VAC_COLUMN_NAME:   TOL_INITIAL,
            }
            for col, tol in self._solver_tol.items():
                self.bridge.set_solver_tolerance(col, tol)
            try:
                self._operating_point = self.bridge.get_current_operating_point()
                logger.info(f"Baseline operating point loaded: {self._operating_point}")
            except Exception as exc:
                logger.warning(f"Could not read operating point: {exc}")

        obs = self._get_observation()
        self._last_state = self._raw_state()
        return obs, {
            "state": self._last_state,
            "action_scales": ACTION_SCALES,
            "operating_point": self._operating_point,
        }

    def step(self, action: np.ndarray):
        self.current_step += 1

        # 1. Denormalize action from [-1, 1] → real ranges
        real_action = self._denormalize_action(action)
        action_dict = dict(zip(ACTION_KEYS, real_action.tolist()))
        self._last_action = action_dict

        # 2. Apply action + solve (with adaptive tolerance)
        if not self.use_mock:
            logger.info(
                f"── Step {self.current_step}/{self.max_steps} ─ "
                f"applying 16 action deltas (warmup={self._warmup_factor():.2f})"
            )
            self.bridge.apply_action(action_dict)
            errors = self._solve_with_tolerance()
            if errors:
                logger.warning(
                    f"Episode {CDUEnvironment._episode_number} END (solver failed) "
                    f"after {self.current_step} steps.\n"
                    f"  Error: {errors}\n"
                    f"  → Flowsheet will be reloaded to baseline at next reset.\n"
                    f"  → A different action sequence will be tried in the next episode."
                )
                obs = self._get_observation()
                return obs, -50.0, True, False, {"error": errors, "step": self.current_step}
        
        # 3. Observe
        obs = self._get_observation()
        state = self._raw_state()
        self._last_state = state

        # 4. Reward
        reward = self._calculate_reward(state, action_dict)
        self._episode_reward += reward
        if reward > self._best_profit:
            self._best_profit = reward

        if not self.use_mock:
            logger.info(
                f"   Step {self.current_step} done │ reward={reward:+.4f} │"
                f" ep_cumulative={self._episode_reward:+.4f} │ best={self._best_profit:+.4f}"
            )

        # 5. Termination
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Safety termination
        if not self.use_mock:
            if (state.get("top_temperature", 0) > settings.MAX_COLUMN_TEMP or
                state.get("bottom_temperature", 0) > settings.MAX_COLUMN_TEMP):
                terminated = True
                reward -= 100.0  # safety penalty

        if not self.use_mock and (terminated or truncated):
            reason = "max steps reached" if truncated else "safety limit exceeded"
            logger.info(
                f"Episode {CDUEnvironment._episode_number} END ({reason}) "
                f"after {self.current_step} steps │ "
                f"total_reward={self._episode_reward:+.4f} │ "
                f"best_ever={self._best_profit:+.4f}\n"
                f"  → Flowsheet will be reloaded to baseline at next reset."
            )

        info = {
            "state": state,
            "action": action_dict,
            "step": self.current_step,
            "episode_reward": self._episode_reward,
            "profit": reward,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.bridge:
            self.bridge.close()

    # ── Helpers ─────────────────────────────────────────────────────────

    def update_prices(self, prices: dict[str, float]) -> None:
        """Live-update product prices (used when user changes prices)."""
        self.prices.update(prices)
        logger.info(f"Prices updated: {self.prices}")

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Map [-1, 1] → per-step delta in physical units, scaled by warmup."""
        scales = np.array([ACTION_SCALES[k] for k in ACTION_KEYS], dtype=np.float32)
        return action * scales * self._warmup_factor()

    def _warmup_factor(self) -> float:
        """Progressive scale: starts at WARMUP_INITIAL, ramps to 1.0."""
        t = self.current_step / max(self.max_steps, 1)
        return min(1.0, WARMUP_INITIAL + (1.0 - WARMUP_INITIAL) * t / WARMUP_END_FRAC)

    def _solve_with_tolerance(self) -> list[str]:
        """Run solver; on failure escalate tolerance and retry once."""
        errors = self.bridge.solve()
        if not errors:
            # Success → slowly tighten tolerances
            for col in self._solver_tol:
                old = self._solver_tol[col]
                new = max(TOL_MIN, old * TOL_RELAX_FACTOR)
                if new != old:
                    self._solver_tol[col] = new
                    self.bridge.set_solver_tolerance(col, new)
            return []

        # First attempt failed → escalate tolerances and retry
        logger.info(
            f"Solver failed — escalating tolerances from "
            f"{self._solver_tol} and retrying"
        )
        for col in self._solver_tol:
            old = self._solver_tol[col]
            new = min(TOL_MAX, old * TOL_ESCALATION_FACTOR)
            self._solver_tol[col] = new
            self.bridge.set_solver_tolerance(col, new)

        errors = self.bridge.solve()
        if not errors:
            logger.info(
                f"Retry succeeded with tolerances {self._solver_tol}"
            )
            return []
        return errors

    def _normalize_action(self, real_delta: np.ndarray) -> np.ndarray:
        """Map physical-unit delta → [-1, 1]."""
        scales = np.array([ACTION_SCALES[k] for k in ACTION_KEYS], dtype=np.float32)
        return real_delta / np.maximum(scales, 1e-8)

    def _raw_state(self) -> dict:
        """Get raw (un-normalized) state from DWSIM or mock."""
        if self.use_mock:
            return self._mock_state()
        return self.bridge.get_column_state()

    def _get_observation(self) -> np.ndarray:
        """Build a 37-dim normalized observation vector."""
        state = self._raw_state()
        obs = np.zeros(37, dtype=np.float32)

        # 13 product flows (indices 0–12)
        for i, prod in enumerate(PRODUCT_KEYS):
            flow = state.get(f"flow_{prod}", 0.0)
            lo, hi = OBS_RANGES["flow"]
            obs[i] = np.clip((flow - lo) / (hi - lo), 0, 1)

        # 13 product temperatures (indices 13–25)
        for i, prod in enumerate(PRODUCT_KEYS):
            temp = state.get(f"temp_{prod}", 0.0)
            lo, hi = OBS_RANGES["temp"]
            obs[13 + i] = np.clip((temp - lo) / (hi - lo), 0, 1)

        # Feed + column state (indices 26–36)
        mappings = [
            ("feed_temperature",       "temp"),
            ("feed_flow_rate",         "feed_flow"),
            ("top_pressure",           "pressure"),
            ("bottom_pressure",        "pressure"),
            ("top_temperature",        "temp"),
            ("bottom_temperature",     "temp"),
            ("condenser_duty",         "duty"),
            ("vac_top_pressure",       "vac_pressure"),
            ("vac_bottom_pressure",    "vac_pressure"),
            ("vac_condenser_duty",     "duty"),
            ("vac_bottom_temperature", "temp"),
        ]
        for j, (key, range_key) in enumerate(mappings):
            val = state.get(key, 0.0)
            lo, hi = OBS_RANGES.get(range_key, (0, 1))
            obs[26 + j] = np.clip((val - lo) / (hi - lo), 0, 1)

        return obs

    def _calculate_reward(self, state: dict, action: dict) -> float:
        """
        Reward = product_revenue - feed_cost - d95_penalty - safety_penalty

        Revenue  = Σ product_flow_rate × product_price  (ADU + NSU + VDU products)
        Feed     = feed_flow_rate × feed_crude_price
        D95%     = penalty for products exceeding their distillation spec
        Safety   = penalty for approaching operational limits
        """
        # ── Product Revenue ──
        revenue = 0.0
        for prod in PRODUCT_KEYS:
            flow = state.get(f"flow_{prod}", 0.0)  # kg/h
            price = self.prices.get(prod, 0.0)       # $/kg
            revenue += flow * price

        # ── Feed Cost ──
        feed_flow = state.get("feed_flow_rate", 0.0)  # kg/h
        feed_price = self.prices.get("Feed_Crude", DEFAULT_PRICES["Feed_Crude"])
        feed_cost = feed_flow * feed_price

        # ── D95% Quality Penalty ──
        d95_penalty = 0.0
        for prod, spec in D95_SPECS.items():
            d95_val = state.get(f"d95_{prod}", 0.0)
            if d95_val > 0 and d95_val > spec:
                violation = d95_val - spec
                d95_penalty += violation * 2.0

        # Safety margin penalty (soft constraint)
        safety_penalty = 0.0
        top_t = state.get("top_temperature", 0.0)
        bot_t = state.get("bottom_temperature", 0.0)
        vac_bot_t = state.get("vac_bottom_temperature", 0.0)
        if top_t > settings.MAX_COLUMN_TEMP * 0.9:
            safety_penalty += (top_t - settings.MAX_COLUMN_TEMP * 0.9) * 2.0
        if bot_t > settings.MAX_COLUMN_TEMP * 0.95:
            safety_penalty += (bot_t - settings.MAX_COLUMN_TEMP * 0.95) * 2.0
        if vac_bot_t > settings.MAX_COLUMN_TEMP * 0.95:
            safety_penalty += (vac_bot_t - settings.MAX_COLUMN_TEMP * 0.95) * 2.0

        profit = revenue - feed_cost - d95_penalty - safety_penalty

        # Scale reward for stable training (keep in reasonable range)
        return profit / 100.0

    def _mock_state(self) -> dict:
        """Generate plausible mock data for testing without DWSIM (ADU + NSU + VDU)."""
        rng = self.np_random if hasattr(self, "np_random") and self.np_random else np.random.default_rng()
        base_flows = {
            "Uncondensed_Gas": 120, "Heavy_Naphtha": 250, "SKO": 500,
            "Light_Gas_Oil": 550, "Heavy_Gas_Oil": 700,
            "StabOffGas": 40, "LPG": 100, "SRN": 350,
            "Offgas": 30, "Vacuum_Diesel": 370, "Vacuum_Gas_Oil": 460,
            "Hotwell_Oil": 130, "Vac_residue": 900,
        }
        base_temps = {
            "Uncondensed_Gas": 50, "Heavy_Naphtha": 155, "SKO": 220,
            "Light_Gas_Oil": 280, "Heavy_Gas_Oil": 340,
            "StabOffGas": 40, "LPG": 45, "SRN": 90,
            "Offgas": 60, "Vacuum_Diesel": 250, "Vacuum_Gas_Oil": 350,
            "Hotwell_Oil": 380, "Vac_residue": 420,
        }
        # Mock D95% values (close to spec limits)
        base_d95 = {
            "Uncondensed_Gas": 0.0, "Heavy_Naphtha": 210.0, "SKO": 280.0,
            "Light_Gas_Oil": 355.0, "Heavy_Gas_Oil": 385.0,
            "StabOffGas": 0.0, "LPG": -5.0, "SRN": 165.0,
            "Offgas": 0.0, "Vacuum_Diesel": 380.0, "Vacuum_Gas_Oil": 500.0,
            "Hotwell_Oil": 450.0, "Vac_residue": 0.0,
        }

        state = {}
        for prod in PRODUCT_KEYS:
            noise = self.curriculum_level * rng.normal(0, 3)
            state[f"flow_{prod}"] = max(0, base_flows[prod] + noise)
            noise_t = self.curriculum_level * rng.normal(0, 5)
            state[f"temp_{prod}"] = base_temps[prod] + noise_t
            noise_d = self.curriculum_level * rng.normal(0, 3)
            state[f"d95_{prod}"] = base_d95[prod] + noise_d if base_d95[prod] != 0.0 else 0.0

        state["top_temperature"] = 50 + rng.normal(0, 2)
        state["bottom_temperature"] = 340 + rng.normal(0, 5)
        state["feed_temperature"] = 399 + rng.normal(0, 3)
        state["feed_flow_rate"] = 4736 + rng.normal(0, 50)
        state["top_pressure"] = 101 + rng.normal(0, 5)
        state["bottom_pressure"] = 116 + rng.normal(0, 5)
        state["condenser_duty"] = 42000 + rng.normal(0, 500)
        state["reboiler_duty"] = 46000 + rng.normal(0, 500)
        # Vacuum column
        state["vac_top_pressure"] = 8 + rng.normal(0, 0.5)
        state["vac_bottom_pressure"] = 15 + rng.normal(0, 1)
        state["vac_condenser_duty"] = 12000 + rng.normal(0, 300)
        state["vac_reboiler_duty"] = 18000 + rng.normal(0, 400)
        state["vac_bottom_temperature"] = 420 + rng.normal(0, 5)

        return state
