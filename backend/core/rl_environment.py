"""
Gymnasium environment for the Crude Distillation Unit (CDU + VDU).

The main_sim.dwxmz flowsheet has two columns:
  1. Atmospheric Distillation Column — produces Uncondensed Gas, USN,
     HN, SKO, LD, HD
  2. Vacuum Distillation Column — receives the atmospheric residue
     and produces Vac Diesel, VGO, Slop Cut, Vac Residue

Observation space (31-dim continuous):
    10 product flow rates + 10 product temperatures +
    feed_temperature + feed_flow + top_pressure + bottom_pressure +
    top_temperature + bottom_temperature + condenser_duty +
    vac_top_pressure + vac_bottom_pressure + vac_condenser_duty +
    vac_bottom_temperature

Action space (10-dim continuous):
    reflux_ratio, usn_draw_temp, hn_draw_temp, sko_draw_temp,
    ld_draw_temp, hd_draw_temp, atmos_steam_rate,
    vac_reflux_ratio, vac_diesel_draw_temp, vgo_draw_temp

Reward = Σ (product_flow × product_price) − energy_penalty − safety_penalty
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional
from loguru import logger

from backend.core.dwsim_bridge import DWSIMBridge
from backend.config import settings


# ── Normalization helpers ───────────────────────────────────────────────────
# These are approximate ranges used to normalize observations to [0, 1].
OBS_RANGES = {
    "flow":        (0.0, 500.0),     # kg/h
    "temp":        (0.0, 500.0),     # °C
    "pressure":    (50.0, 400.0),    # kPa
    "vac_pressure":(1.0, 50.0),      # kPa (vacuum side)
    "feed_flow":   (0.0, 5000.0),    # kg/h
    "duty":        (0.0, 50_000.0),  # kW
}

# Action ranges (low, high) — 10 dimensions: 7 ADU + 3 VDU
ACTION_RANGES = np.array([
    # --- Atmospheric column ---
    [0.5,   8.0],     # reflux_ratio
    [60.0,  180.0],   # usn_draw_temp (naphtha)
    [120.0, 220.0],   # hn_draw_temp
    [170.0, 280.0],   # sko_draw_temp
    [240.0, 340.0],   # ld_draw_temp
    [300.0, 380.0],   # hd_draw_temp
    [0.0,   5000.0],  # atmos_steam_rate
    # --- Vacuum column ---
    [0.5,   5.0],     # vac_reflux_ratio
    [200.0, 350.0],   # vac_diesel_draw_temp
    [300.0, 420.0],   # vgo_draw_temp
], dtype=np.float32)

ACTION_KEYS = [
    "reflux_ratio", "usn_draw_temp", "hn_draw_temp",
    "sko_draw_temp", "ld_draw_temp", "hd_draw_temp",
    "atmos_steam_rate",
    "vac_reflux_ratio", "vac_diesel_draw_temp", "vgo_draw_temp",
]

PRODUCT_KEYS = [
    "Uncondensed_Gas", "USN", "HN", "SKO", "LD", "HD",
    "Vac_Diesel", "VGO", "Slop_Cut", "Vac_Residue",
]

# Default product prices ($/kg) — will be overridden by user prices
DEFAULT_PRICES = {
    "Uncondensed_Gas": 0.40,
    "USN":             0.72,   # naphtha (+ LPG simplified)
    "HN":              0.68,
    "SKO":             0.75,   # jet fuel
    "LD":              0.60,
    "HD":              0.55,
    "Vac_Diesel":      0.52,
    "VGO":             0.45,
    "Slop_Cut":        0.30,
    "Vac_Residue":     0.25,
}

# Energy cost coefficient ($/kW·h)
ENERGY_COST = 0.05


class CDUEnvironment(gym.Env):
    """
    Gymnasium environment wrapping a DWSIM CDU flowsheet.

    The agent learns to set column operating parameters to maximize
    profit = revenue(product flows × prices) − energy cost − penalties.
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

        # Observation: 31-dimensional (10 flows + 10 temps + 11 column state)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(31,), dtype=np.float32
        )

        # Action: 10-dimensional, normalized to [-1, 1] (SB3 convention)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # DWSIM bridge
        if not use_mock:
            self.bridge = DWSIMBridge(flowsheet_path)
        else:
            self.bridge = None

        # Episode tracking
        self._episode_reward = 0.0
        self._best_profit = float("-inf")
        self._last_state: Optional[dict] = None
        self._last_action: Optional[dict] = None

        logger.info(
            f"CDUEnvironment created | mock={use_mock} | "
            f"max_steps={max_steps} | curriculum={curriculum_level:.2f}"
        )

    # ── Gym API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._episode_reward = 0.0

        if not self.use_mock:
            self.bridge.load()

        obs = self._get_observation()
        self._last_state = self._raw_state()
        return obs, {"state": self._last_state}

    def step(self, action: np.ndarray):
        self.current_step += 1

        # 1. Denormalize action from [-1, 1] → real ranges
        real_action = self._denormalize_action(action)
        action_dict = dict(zip(ACTION_KEYS, real_action.tolist()))
        self._last_action = action_dict

        # 2. Apply action + solve
        if not self.use_mock:
            self.bridge.apply_action(action_dict)
            errors = self.bridge.solve()
            if errors:
                obs = self._get_observation()
                return obs, -50.0, True, False, {"error": errors}
        
        # 3. Observe
        obs = self._get_observation()
        state = self._raw_state()
        self._last_state = state

        # 4. Reward
        reward = self._calculate_reward(state, action_dict)
        self._episode_reward += reward
        if reward > self._best_profit:
            self._best_profit = reward

        # 5. Termination
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Safety termination
        if not self.use_mock:
            if (state.get("top_temperature", 0) > settings.MAX_COLUMN_TEMP or
                state.get("bottom_temperature", 0) > settings.MAX_COLUMN_TEMP):
                terminated = True
                reward -= 100.0  # safety penalty

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
        """Map [-1, 1] → real action ranges."""
        low = ACTION_RANGES[:, 0]
        high = ACTION_RANGES[:, 1]
        return low + (action + 1.0) / 2.0 * (high - low)

    def _normalize_action(self, real_action: np.ndarray) -> np.ndarray:
        """Map real action ranges → [-1, 1]."""
        low = ACTION_RANGES[:, 0]
        high = ACTION_RANGES[:, 1]
        return 2.0 * (real_action - low) / (high - low) - 1.0

    def _raw_state(self) -> dict:
        """Get raw (un-normalized) state from DWSIM or mock."""
        if self.use_mock:
            return self._mock_state()
        return self.bridge.get_column_state()

    def _get_observation(self) -> np.ndarray:
        """Build a 31-dim normalized observation vector."""
        state = self._raw_state()
        obs = np.zeros(31, dtype=np.float32)

        # 10 product flows (indices 0–9)
        for i, prod in enumerate(PRODUCT_KEYS):
            flow = state.get(f"flow_{prod}", 0.0)
            lo, hi = OBS_RANGES["flow"]
            obs[i] = np.clip((flow - lo) / (hi - lo), 0, 1)

        # 10 product temperatures (indices 10–19)
        for i, prod in enumerate(PRODUCT_KEYS):
            temp = state.get(f"temp_{prod}", 0.0)
            lo, hi = OBS_RANGES["temp"]
            obs[10 + i] = np.clip((temp - lo) / (hi - lo), 0, 1)

        # Feed + column state (indices 20–30)
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
            obs[20 + j] = np.clip((val - lo) / (hi - lo), 0, 1)

        return obs

    def _calculate_reward(self, state: dict, action: dict) -> float:
        """
        Reward = revenue − energy_cost − safety_penalty

        Revenue = Σ product_flow_rate × product_price  (ADU + VDU products)
        Energy  = (reboiler_duty + condenser_duty) × cost_coeff  (both columns)
        Safety  = penalty for approaching operational limits
        """
        # Revenue
        revenue = 0.0
        for prod in PRODUCT_KEYS:
            flow = state.get(f"flow_{prod}", 0.0)  # kg/h
            price = self.prices.get(prod, 0.0)       # $/kg
            revenue += flow * price

        # Energy cost (both columns)
        atmos_reboiler = abs(state.get("reboiler_duty", 0.0))
        atmos_condenser = abs(state.get("condenser_duty", 0.0))
        vac_reboiler = abs(state.get("vac_reboiler_duty", 0.0))
        vac_condenser = abs(state.get("vac_condenser_duty", 0.0))
        energy_cost = (atmos_reboiler + atmos_condenser + vac_reboiler + vac_condenser) * ENERGY_COST

        # Reflux penalty (encourages efficiency on both columns)
        reflux = action.get("reflux_ratio", 1.0)
        vac_reflux = action.get("vac_reflux_ratio", 1.0)
        reflux_penalty = max(0, reflux - 3.0) * 10.0
        reflux_penalty += max(0, vac_reflux - 2.5) * 10.0

        # Safety margin penalty (soft constraint)
        safety_penalty = 0.0
        top_t = state.get("top_temperature", 0.0)
        bot_t = state.get("bottom_temperature", 0.0)
        vac_bot_t = state.get("vac_bottom_temperature", 0.0)
        if top_t > settings.MAX_COLUMN_TEMP * 0.9:
            safety_penalty += (top_t - settings.MAX_COLUMN_TEMP * 0.9) * 5.0
        if bot_t > settings.MAX_COLUMN_TEMP * 0.95:
            safety_penalty += (bot_t - settings.MAX_COLUMN_TEMP * 0.95) * 5.0
        if vac_bot_t > settings.MAX_COLUMN_TEMP * 0.95:
            safety_penalty += (vac_bot_t - settings.MAX_COLUMN_TEMP * 0.95) * 5.0

        reward = revenue - energy_cost - reflux_penalty - safety_penalty

        # Scale reward for stable training
        return reward / 1000.0

    def _mock_state(self) -> dict:
        """Generate plausible mock data for testing without DWSIM (ADU + VDU)."""
        rng = self.np_random if hasattr(self, "np_random") and self.np_random else np.random.default_rng()
        # Flows in kg/h (matching DWSIM bridge output)
        base_flows = {
            "Uncondensed_Gas": 25, "USN": 106, "HN": 38, "SKO": 43,
            "LD": 51, "HD": 69,
            "Vac_Diesel": 32, "VGO": 45, "Slop_Cut": 12, "Vac_Residue": 78,
        }
        base_temps = {
            "Uncondensed_Gas": 50, "USN": 90, "HN": 155, "SKO": 220,
            "LD": 280, "HD": 340,
            "Vac_Diesel": 250, "VGO": 350, "Slop_Cut": 380, "Vac_Residue": 420,
        }

        state = {}
        for prod in PRODUCT_KEYS:
            noise = self.curriculum_level * rng.normal(0, 3)
            state[f"flow_{prod}"] = max(0, base_flows[prod] + noise)
            noise_t = self.curriculum_level * rng.normal(0, 5)
            state[f"temp_{prod}"] = base_temps[prod] + noise_t

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
