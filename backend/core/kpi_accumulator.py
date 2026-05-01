"""
KPI Accumulator: Persists run-wise metrics (distillate yields, energy savings, profit)
to a local JSON file. Called after each optimization run.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger

KPI_FILE = str(Path(__file__).parent.parent / "data" / "kpi_accumulator.json")


class KPIAccumulator:
    """Manages persistent accumulation of KPI metrics across all optimization runs."""

    @staticmethod
    def _empty_data() -> dict:
        return {
            "runs": [],
            "summary": {
                "total_runs": 0,
                "positive_profit_runs": 0,
                "avg_distillate_yield_improvement": 0.0,
                "cumulative_energy_savings": 0.0,
                "cumulative_profit": 0.0,
            },
        }

    @staticmethod
    def _num(value: Optional[float]) -> float:
        """Convert numeric-like values (including numpy scalars) to plain float."""
        try:
            if value is None:
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def load() -> dict:
        """Load existing KPI data from JSON file, or return empty structure."""
        if os.path.exists(KPI_FILE):
            try:
                with open(KPI_FILE, "r") as f:
                    data = json.load(f)
                    if "runs" not in data:
                        data["runs"] = []
                    if "summary" not in data or not isinstance(data["summary"], dict):
                        data["summary"] = KPIAccumulator._empty_data()["summary"]
                    return data
            except Exception as exc:
                logger.warning(f"Failed to load KPI file: {exc}, starting fresh")
        return KPIAccumulator._empty_data()

    @staticmethod
    def save(data: dict) -> None:
        """Save KPI data to JSON file."""
        try:
            os.makedirs(os.path.dirname(KPI_FILE), exist_ok=True)
            with open(KPI_FILE, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"KPI data saved to {KPI_FILE}")
        except Exception as exc:
            logger.error(f"Failed to save KPI file: {exc}")

    @staticmethod
    def reset() -> None:
        """Reset KPI data to empty summary and save it."""
        data = KPIAccumulator._empty_data()
        KPIAccumulator.save(data)

    @staticmethod
    def record_run(
        base_state: dict,
        opt_state: dict,
        base_profit: float,
        opt_profit: float,
        reboiler_duty_base: Optional[float] = None,
        reboiler_duty_opt: Optional[float] = None,
        furnace_duty_base: Optional[float] = None,
        furnace_duty_opt: Optional[float] = None,
    ) -> None:
        """
        Record a completed optimization run.

        Metrics calculated:
          - Distillate yield improvement: Σ(product flows) for key distillates
          - Energy savings: Reduction in reboiler + furnace duties
          - Profit: Only positive profits accumulated (skip negative)
        """
        data = KPIAccumulator.load()

        # Key distillate products (high-value products)
        distillate_products = [
            "Heavy_Naphtha", "SKO", "Light_Gas_Oil", "Heavy_Gas_Oil",
            "Vacuum_Diesel", "Vacuum_Gas_Oil"
        ]

        # Calculate distillate yield
        base_yield = sum(KPIAccumulator._num(base_state.get(f"flow_{p}", 0.0)) for p in distillate_products)
        opt_yield = sum(KPIAccumulator._num(opt_state.get(f"flow_{p}", 0.0)) for p in distillate_products)
        distillate_improvement = opt_yield - base_yield

        # Calculate energy savings (lower duty = positive savings; negative means more energy used → 0)
        energy_savings = 0.0
        if reboiler_duty_base is not None and reboiler_duty_opt is not None:
            energy_savings += KPIAccumulator._num(reboiler_duty_base) - KPIAccumulator._num(reboiler_duty_opt)
        if furnace_duty_base is not None and furnace_duty_opt is not None:
            energy_savings += KPIAccumulator._num(furnace_duty_base) - KPIAccumulator._num(furnace_duty_opt)
        energy_savings = max(0.0, energy_savings)

        # Profit benefit from optimization (only positive deltas count)
        profit_delta = KPIAccumulator._num(opt_profit) - KPIAccumulator._num(base_profit)
        profit_generated = max(0.0, profit_delta)

        run = {
            "timestamp": datetime.utcnow().isoformat(),
            "distillate_yield_improvement": round(distillate_improvement, 2),  # kg/h
            "energy_savings": round(energy_savings, 2),  # kW
            "profit_generated": round(profit_generated, 2),  # $/h
            "profit_delta": round(profit_delta, 2),  # $/h
            "base_profit": round(KPIAccumulator._num(base_profit), 2),
            "opt_profit": round(KPIAccumulator._num(opt_profit), 2),
        }

        data["runs"].append(run)
        KPIAccumulator._update_summary(data)
        KPIAccumulator.save(data)

    @staticmethod
    def _update_summary(data: dict) -> None:
        """Calculate summary statistics from all runs."""
        runs = data.get("runs", [])
        if not runs:
            data["summary"] = KPIAccumulator._empty_data()["summary"]
            return

        # Normalize legacy rows to keep summary math consistent across versions.
        for run in runs:
            base_profit = KPIAccumulator._num(run.get("base_profit", 0.0))
            opt_profit = KPIAccumulator._num(run.get("opt_profit", 0.0))
            delta = opt_profit - base_profit
            run["profit_delta"] = round(delta, 2)
            run["profit_generated"] = round(max(0.0, delta), 2)

        # Filter positive-profit-benefit runs only
        positive_runs = [
            r for r in runs
            if max(
                0.0,
                KPIAccumulator._num(r.get("opt_profit", 0.0)) - KPIAccumulator._num(r.get("base_profit", 0.0)),
            ) > 0
        ]

        if positive_runs:
            avg_yield = sum(r["distillate_yield_improvement"] for r in positive_runs) / len(positive_runs)
            cumulative_energy = sum(max(0.0, r["energy_savings"]) for r in positive_runs)
            total_profit = sum(
                max(
                    0.0,
                    KPIAccumulator._num(r.get("opt_profit", 0.0)) - KPIAccumulator._num(r.get("base_profit", 0.0)),
                )
                for r in positive_runs
            )
        else:
            avg_yield = 0.0
            cumulative_energy = 0.0
            total_profit = 0.0

        data["summary"] = {
            "total_runs": len(runs),
            "positive_profit_runs": len(positive_runs),
            "avg_distillate_yield_improvement": round(avg_yield, 2),  # kg/h
            "cumulative_energy_savings": round(cumulative_energy, 2),  # kW
            "cumulative_profit": round(total_profit, 2),  # $/h (sum across all positive runs)
        }

    @staticmethod
    def get_stats() -> dict:
        """Get current KPI statistics."""
        data = KPIAccumulator.load()
        KPIAccumulator._update_summary(data)
        KPIAccumulator.save(data)
        return {
            "summary": data.get("summary", KPIAccumulator._empty_data()["summary"]),
            "run_count": len(data.get("runs", [])),
            "latest_runs": data.get("runs", [])[-5:],  # Last 5 runs
        }
