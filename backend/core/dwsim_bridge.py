"""
DWSIM Automation Bridge — wraps the DWSIM COM/.NET automation API
so the rest of the backend never touches CLR directly.

The model uses main_sim.dwxmz which contains three columns:
  1. Atmospheric Distillation Column — object name: "Atmos_Column"
  2. Naphtha Stabilizer              — object name: "Naphtha_Stabilizer"
  3. Vacuum Distillation Column      — object name: "Vacuum_Column"

Major product streams:
  ADU     : Uncondensed_Gas, Heavy_Naphtha, SKO, Light_Gas_Oil, Heavy_Gas_Oil
  NSU     : StabOffGas, LPG, SRN
  VDU     : Offgas, Vacuum_Diesel, Vacuum_Gas_Oil, Hotwell_Oil, Vac_residue

If the model uses the PR/LK property package, it is automatically
patched to use standard Peng-Robinson (PR) on first load.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import zipfile
from typing import Any, Optional
from loguru import logger

from backend.config import settings

# ── .NET / CLR bootstrap ────────────────────────────────────────────────────
_clr_lock = threading.Lock()
_clr_ready = False


def _bootstrap_clr() -> None:
    """Add DWSIM assemblies to the CLR; safe to call more than once."""
    global _clr_ready
    if _clr_ready:
        return
    with _clr_lock:
        if _clr_ready:
            return
        dwsim = settings.DWSIM_PATH
        sys.path.append(dwsim)
        import clr  # type: ignore

        for dll in [
            "CapeOpen.dll",
            "DWSIM.Automation.dll",
            "DWSIM.Interfaces.dll",
            "DWSIM.GlobalSettings.dll",
            "DWSIM.SharedClasses.dll",
            "DWSIM.Thermodynamics.dll",
            "DWSIM.UnitOperations.dll",
            "DWSIM.Inspector.dll",
        ]:
            path = os.path.join(dwsim, dll)
            if os.path.exists(path):
                clr.AddReference(path)
        _clr_ready = True
        logger.info("CLR bootstrapped with DWSIM assemblies")


# ── CDU + NSU + VDU object-name mapping ────────────────────────────────────
# These match the main_sim.dwxmz flowsheet exactly.

# Atmospheric column
ATMOS_COLUMN_NAME  = "Atmos_Column"
FEED_STREAM        = "Crude_Feed"                       # preheated crude into Atmos_Column
FEED_INPUT         = "Erha"                       # raw crude inlet (before furnace)
FURNACE            = "Heater"                            # atmospheric furnace


# Naphtha Stabilizer
NSU_COLUMN_NAME    = "Naphtha_Stabilizer"

# Vacuum column
VAC_COLUMN_NAME    = "Vacuum_Column"
VAC_FEED_STREAM    = "Vac_Feed"                          # heated RCO entering Vacuum_Column
VAC_RCO_STREAM     = "Reduced_Crude_Oil"                 # RCO from atmos bottom → pump
VAC_FURNACE        = "Vac_Furnace"                       # vacuum heater

# Legacy aliases
COLUMN_NAME  = ATMOS_COLUMN_NAME

# ── Product streams (ADU + NSU + VDU) ──────────────────────────────────────
PRODUCT_STREAMS: dict[str, str] = {
    # --- Atmospheric products ---
    "Uncondensed_Gas": "Uncondensed_Gas",       # overhead off-gas
    "Heavy_Naphtha":   "Heavy_Naphtha",          # heavy naphtha side draw (→ NSU feed)
    "SKO":             "SKO",                    # jet fuel / kerosene
    "Light_Gas_Oil":   "Light_Gas_Oil",          # light diesel
    "Heavy_Gas_Oil":   "Heavy_Gas_Oil",          # heavy diesel
    # --- Naphtha Stabilizer products ---
    "StabOffGas":      "StabOffGas",             # stabilizer overhead off-gas
    "LPG":             "LPG",                    # liquefied petroleum gas
    "SRN":             "SRN",                    # stabilized / straight-run naphtha
    # --- Vacuum products ---
    "Offgas":          "Offgas",                 # VDU overhead gas
    "Vacuum_Diesel":   "Vacuum_Diesel",          # VDU side-draw diesel
    "Vacuum_Gas_Oil":  "Vacuum_Gas_Oil",         # vacuum gas oil
    "Hotwell_Oil":     "Hotwell_Oil",            # hotwell sump / slop cut
    "Vac_residue":     "Vac_residue",            # vacuum residue (bottom)
}

# ── DWSIM property identifiers (source: dwsim.org/wiki Object_Property_Codes) ──
# Material Streams:
#   PROP_MS_0  = Temperature (K)
#   PROP_MS_1  = Pressure    (Pa)
#   PROP_MS_2  = Mass Flow   (kg/s)      ← correct per DWSIM wiki
#   PROP_MS_3  = Molar Flow  (mol/s)
#   PROP_MS_4  = Volumetric Flow (m³/s)  ← NOT mass flow!
#
# Distillation Column:
#   PROP_DC_0  = Condenser Pressure (Pa)
#   PROP_DC_1  = Reboiler Pressure  (Pa)
#   PROP_DC_2  = Condenser Pressure Drop (Pa)  — NOT reflux ratio!
#   PROP_DC_3  = Reflux Ratio (read-only calculated value)
#   PROP_DC_4  = Distillate Molar Flow
#   PROP_DC_5  = Condenser Duty (W)
#   PROP_DC_6  = Reboiler Duty  (W)
#   PROP_DC_7  = Number of Stages
#
# NOTE: Reflux ratio must be read/written via .NET reflection on
#       column.Specs["C"].SpecValue — PROP_DC_3 is read-only.


def _patch_model_if_needed(src_path: str) -> str:
    """
    If the model uses the PR/LK property package (unavailable in
    some DWSIM installs), create a patched copy using standard PR.
    Returns the path to use (original or patched).
    """
    if not os.path.exists(src_path):
        return src_path

    # Quick check: does the XML contain PengRobinsonLKPropertyPackage?
    try:
        with zipfile.ZipFile(src_path, "r") as z:
            xml_name = [n for n in z.namelist() if n.endswith(".xml")][0]
            with z.open(xml_name) as f:
                xml_bytes = f.read()
    except Exception:
        return src_path

    if b"PengRobinsonLKPropertyPackage" not in xml_bytes:
        return src_path  # no patching needed

    # Create patched version alongside the original
    base, ext = os.path.splitext(src_path)
    patched = base + "_patched" + ext
    if os.path.exists(patched):
        logger.info(f"Using existing patched model: {patched}")
        return patched

    logger.warning("Model uses PR/LK property package — patching to standard PR")
    tmp_dir = src_path + "_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(src_path, "r") as z:
            z.extractall(tmp_dir)
        for fname in os.listdir(tmp_dir):
            if fname.endswith(".xml"):
                fpath = os.path.join(tmp_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    xml = f.read()
                xml = xml.replace(
                    "PengRobinsonLKPropertyPackage",
                    "PengRobinsonPropertyPackage",
                )
                xml = xml.replace(
                    "Peng-Robinson / Lee-Kesler (PR/LK)", "Peng-Robinson (PR)"
                )
                xml = xml.replace(
                    "Lee-Kesler (PR/LK)", "Peng-Robinson (PR)"
                )
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(xml)
        with zipfile.ZipFile(patched, "w", zipfile.ZIP_DEFLATED) as z:
            for fname in os.listdir(tmp_dir):
                z.write(os.path.join(tmp_dir, fname), fname)
        logger.info(f"Patched model saved: {patched}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return patched


# ── Hard process limits for incremental action clamping ─────────────────────
# These prevent multi-step delta accumulation from drifting outside the
# Newton-iteration convergence basin of the DWSIM column solver.
_ACTION_HARD_LIMITS: dict[str, tuple[float, float]] = {
    "reflux_ratio":         (1.0,   12.0),
    "hn_draw_temp":         (100.0, 280.0),   # °C
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


class DWSIMBridge:
    """Thin, thread-safe wrapper around the DWSIM Automation3 interface."""

    def __init__(self, flowsheet_path: Optional[str] = None):
        _bootstrap_clr()
        from DWSIM.Automation import Automation3  # type: ignore

        self.interf = Automation3()
        raw_path = flowsheet_path or settings.FLOWSHEET_PATH
        # Make absolute
        if not os.path.isabs(raw_path):
            # Resolve relative to project root (parent of backend/)
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            raw_path = os.path.join(project_root, raw_path)
        self.flowsheet_path = _patch_model_if_needed(raw_path)
        self.sim = None
        self._lock = threading.Lock()
        logger.info(f"DWSIMBridge created for {self.flowsheet_path}")

    # ── Lifecycle ───────────────────────────────────────────────────────

    def load(self) -> None:
        """Load (or reload) the flowsheet."""
        with self._lock:
            self.sim = self.interf.LoadFlowsheet(self.flowsheet_path)
            logger.info("Flowsheet loaded")

    # ── Solver tolerance management ─────────────────────────────────────

    def get_solver_tolerance(self, column_name: str) -> float:
        """Read the outer-loop convergence tolerance of a column."""
        try:
            col_obj = self._obj(column_name)
            tol = self._reflect(col_obj, "ExternalLoopTolerance")
            return float(tol) if tol is not None else 1.0
        except Exception:
            return 1.0

    def set_solver_tolerance(self, column_name: str, tolerance: float) -> None:
        """Set the outer-loop convergence tolerance of a column (range 0.1–100)."""
        tolerance = max(0.1, min(100.0, tolerance))
        try:
            col_obj = self._obj(column_name)
            prop = col_obj.GetType().GetProperty("ExternalLoopTolerance")
            if prop is not None:
                prop.SetValue(col_obj, float(tolerance), None)
                logger.debug(f"Solver tolerance for {column_name} set to {tolerance}")
            else:
                logger.debug(f"ExternalLoopTolerance property not found on {column_name}")
        except Exception as exc:
            logger.warning(f"Cannot set solver tolerance for {column_name}: {exc}")

    def save(self, path: Optional[str] = None) -> None:
        with self._lock:
            target = path or self.flowsheet_path
            self.interf.SaveFlowsheet2(self.sim, target)
            logger.info(f"Flowsheet saved to {target}")

    def close(self) -> None:
        with self._lock:
            if self.sim is not None:
                self.interf.ReleaseResources()
                self.sim = None
                logger.info("DWSIM resources released")

    # ── Solve ───────────────────────────────────────────────────────────

    def solve(self) -> list[str]:
        """Run the solver. Returns a list of error messages (empty = success)."""
        with self._lock:
            errors = self.interf.CalculateFlowsheet4(self.sim)
            err_list = list(errors) if errors else []
            if err_list:
                logger.warning(f"Solver errors: {err_list}")
            return err_list

    # ── Generic read / write helpers ────────────────────────────────────

    def _obj(self, name: str) -> Any:
        return self.sim.GetFlowsheetSimulationObject(name)

    @staticmethod
    def _reflect(obj, prop_name: str):
        """Access a .NET property via reflection (bypasses ISimulationObject interface)."""
        prop = obj.GetType().GetProperty(prop_name)
        if prop is None:
            return None
        return prop.GetValue(obj, None)

    def get_property(self, obj_name: str, prop_id: str) -> float:
        """Read a scalar property from a flowsheet object using PROP_XX_N ids."""
        with self._lock:
            val = self._obj(obj_name).GetPropertyValue(prop_id)
            if val is None or str(val).strip() == "":
                return 0.0
            return float(val)

    def set_property(self, obj_name: str, prop_id: str, value: float) -> None:
        """Write a scalar property to a flowsheet object."""
        with self._lock:
            self._obj(obj_name).SetPropertyValue(prop_id, value)

    # ── High-level CDU reads ────────────────────────────────────────────

    def get_product_flows(self) -> dict[str, float]:
        """Return mass flow rates (kg/h) for every product."""
        flows: dict[str, float] = {}
        for prod, stream in PRODUCT_STREAMS.items():
            try:
                mass_flow = self.get_property(stream, "PROP_MS_2")
                flows[prod] = mass_flow * 3600  # kg/s → kg/h
            except Exception as exc:
                logger.error(f"Cannot read flow for {prod}: {exc}")
                flows[prod] = 0.0
        return flows

    def get_product_temperatures(self) -> dict[str, float]:
        """Return temperature (°C) for every product stream."""
        temps: dict[str, float] = {}
        for prod, stream in PRODUCT_STREAMS.items():
            try:
                temps[prod] = self.get_property(stream, "PROP_MS_0") - 273.15
            except Exception:
                temps[prod] = 0.0
        return temps

    def get_column_state(self) -> dict[str, Any]:
        """
        Aggregate snapshot used as the RL observation.
        Returns a flat dict of floats covering ADU, NSU, and VDU.
        """
        flows = self.get_product_flows()
        temps = self.get_product_temperatures()

        # --- Atmospheric column ---
        try:
            top_temp = self.get_property(PRODUCT_STREAMS["Uncondensed_Gas"], "PROP_MS_0") - 273.15
        except Exception:
            top_temp = 0.0

        try:
            atmos_bot_temp = self.get_property(VAC_RCO_STREAM, "PROP_MS_0") - 273.15
        except Exception:
            atmos_bot_temp = 0.0

        try:
            feed_temp = self.get_property(FEED_STREAM, "PROP_MS_0") - 273.15
        except Exception:
            feed_temp = 0.0

        try:
            feed_flow = self.get_property(FEED_STREAM, "PROP_MS_2") * 3600  # kg/h
        except Exception:
            feed_flow = 0.0

        try:
            top_press = self.get_property(ATMOS_COLUMN_NAME, "PROP_DC_0") / 1000  # Pa → kPa
        except Exception:
            top_press = 0.0

        try:
            bot_press = self.get_property(ATMOS_COLUMN_NAME, "PROP_DC_1") / 1000  # Pa → kPa
        except Exception:
            bot_press = 0.0

        try:
            condenser_duty = self.get_property(ATMOS_COLUMN_NAME, "PROP_DC_5") / 1000  # W → kW
        except Exception:
            condenser_duty = 0.0

        try:
            atmos_reboiler_duty = self.get_property(ATMOS_COLUMN_NAME, "PROP_DC_6") / 1000  # W → kW
        except Exception:
            atmos_reboiler_duty = 0.0

        # --- Vacuum column ---
        try:
            vac_top_press = self.get_property(VAC_COLUMN_NAME, "PROP_DC_0") / 1000
        except Exception:
            vac_top_press = 0.0

        try:
            vac_bot_press = self.get_property(VAC_COLUMN_NAME, "PROP_DC_1") / 1000
        except Exception:
            vac_bot_press = 0.0

        try:
            vac_condenser_duty = self.get_property(VAC_COLUMN_NAME, "PROP_DC_5") / 1000
        except Exception:
            vac_condenser_duty = 0.0

        try:
            vac_reboiler_duty = self.get_property(VAC_COLUMN_NAME, "PROP_DC_6") / 1000
        except Exception:
            vac_reboiler_duty = 0.0

        try:
            vac_bot_temp = self.get_property(PRODUCT_STREAMS["Vac_residue"], "PROP_MS_0") - 273.15
        except Exception:
            vac_bot_temp = 0.0

        # --- D95% estimates for all products ---
        d95 = self.get_d95_all_products()

        return {
            **{f"flow_{k}": v for k, v in flows.items()},
            **{f"temp_{k}": v for k, v in temps.items()},
            **{f"d95_{k}": v for k, v in d95.items()},
            "top_temperature": top_temp,
            "bottom_temperature": atmos_bot_temp,
            "feed_temperature": feed_temp,
            "feed_flow_rate": feed_flow,
            "top_pressure": top_press,
            "bottom_pressure": bot_press,
            "condenser_duty": condenser_duty,
            "reboiler_duty": atmos_reboiler_duty,
            # Vacuum column state
            "vac_top_pressure": vac_top_press,
            "vac_bottom_pressure": vac_bot_press,
            "vac_condenser_duty": vac_condenser_duty,
            "vac_reboiler_duty": vac_reboiler_duty,
            "vac_bottom_temperature": vac_bot_temp,
        }

    # ── D95% estimation from composition ────────────────────────────────

    def get_stream_composition(self, stream_name: str) -> list[tuple[str, float, float]]:
        """
        Read the overall-phase composition of a material stream.

        Returns a list of (component_name, mass_fraction, normal_boiling_point_°C).
        Accesses the DWSIM .NET object model:
            stream.Phases[0].Compounds → {name: ICompound}
            ICompound.MassFraction, ICompound.ConstantProperties.Normal_Boiling_Point (K)
        """
        result: list[tuple[str, float, float]] = []
        try:
            stream_obj = self._obj(stream_name)
            # Use .NET reflection to bypass ISimulationObject interface
            phases = self._reflect(stream_obj, "Phases")
            p0 = phases[0]  # overall (mixed) phase
            compounds = self._reflect(p0, "Compounds")
            for comp_name in compounds.Keys:
                compound = compounds[comp_name]
                mass_frac = float(self._reflect(compound, "MassFraction") or 0.0)
                const_props = self._reflect(compound, "ConstantProperties")
                nbp_k = float(self._reflect(const_props, "Normal_Boiling_Point") or 0.0)
                if mass_frac > 1e-9:
                    result.append((comp_name, mass_frac, nbp_k - 273.15))
        except Exception as exc:
            logger.debug(f"Cannot read composition for {stream_name}: {exc}")
        return result

    def estimate_d95(self, stream_name: str) -> float:
        """
        Estimate the 95% distillation temperature (D95%) for a product stream.

        Method:
          1. Read each component's mass fraction and Normal Boiling Point.
          2. Sort components by ascending NBP.
          3. Walk the cumulative mass fraction curve.
          4. Linearly interpolate to find the temperature where
             cumulative mass fraction = 0.95.

        Returns D95% in °C, or 0.0 if composition is unavailable.
        """
        comp = self.get_stream_composition(stream_name)
        if not comp:
            return 0.0

        # Sort by Normal Boiling Point (ascending)
        comp.sort(key=lambda x: x[2])

        cumulative = 0.0
        prev_cum = 0.0
        prev_nbp = comp[0][2]
        for _name, mf, nbp in comp:
            prev_cum = cumulative
            cumulative += mf
            if cumulative >= 0.95:
                # Linear interpolation between prev and current NBP
                if cumulative - prev_cum > 1e-12:
                    frac_needed = 0.95 - prev_cum
                    frac_of_bin = frac_needed / (cumulative - prev_cum)
                    return prev_nbp + frac_of_bin * (nbp - prev_nbp)
                return nbp
            prev_nbp = nbp

        # If total mass fraction < 0.95 (e.g. trace components), return last NBP
        return comp[-1][2] if comp else 0.0

    def get_d95_all_products(self) -> dict[str, float]:
        """Get D95% estimates (°C) for all product streams."""
        d95: dict[str, float] = {}
        for prod, stream in PRODUCT_STREAMS.items():
            d95[prod] = self.estimate_d95(stream)
        return d95

    # ── High-level CDU writes ───────────────────────────────────────────

    def get_current_operating_point(self) -> dict[str, float]:
        """
        Read the simulation's current values for every RL-controllable
        parameter.  Returns a dict keyed by ACTION_KEYS with real-unit
        values (°C for temps, kg/h for steam, dimensionless for reflux).

        This is the *baseline* the RL agent should explore around.
        """
        point: dict[str, float] = {}

        # --- Atmospheric column ---
        try:
            col_obj = self._obj(ATMOS_COLUMN_NAME)
            specs = self._reflect(col_obj, "Specs")
            point["reflux_ratio"] = float(self._reflect(specs["C"], "SpecValue"))
        except Exception:
            point["reflux_ratio"] = 5.0  # safe default

        temp_map = {
            "hn_draw_temp":  PRODUCT_STREAMS["Heavy_Naphtha"],
            "sko_draw_temp": PRODUCT_STREAMS["SKO"],
            "ld_draw_temp":  PRODUCT_STREAMS["Light_Gas_Oil"],
            "hd_draw_temp":  PRODUCT_STREAMS["Heavy_Gas_Oil"],
        }
        for key, stream in temp_map.items():
            try:
                point[key] = self.get_property(stream, "PROP_MS_0") - 273.15  # K → °C
            except Exception:
                point[key] = 0.0

        try:
            col_obj = self._obj(ATMOS_COLUMN_NAME)
            atmos_specs = self._reflect(col_obj, "Specs")
            point["atmos_reboiler_temp"] = float(self._reflect(atmos_specs["R"], "SpecValue"))
        except Exception:
            point["atmos_reboiler_temp"] = 365.0

        # --- Naphtha Stabilizer ---
        try:
            nsu_obj = self._obj(NSU_COLUMN_NAME)
            nsu_specs = self._reflect(nsu_obj, "Specs")
            point["nsu_reflux_ratio"] = float(self._reflect(nsu_specs["C"], "SpecValue"))
        except Exception:
            point["nsu_reflux_ratio"] = 5.0

        try:
            nsu_specs2 = self._reflect(self._obj(NSU_COLUMN_NAME), "Specs")
            point["nsu_reboiler_temp"] = float(self._reflect(nsu_specs2["R"], "SpecValue"))
        except Exception:
            point["nsu_reboiler_temp"] = 155.0

        # --- Vacuum column ---
        try:
            vac_obj = self._obj(VAC_COLUMN_NAME)
            vac_specs = self._reflect(vac_obj, "Specs")
            point["vac_reflux_ratio"] = float(self._reflect(vac_specs["C"], "SpecValue"))
        except Exception:
            point["vac_reflux_ratio"] = 5.0

        try:
            vac_specs2 = self._reflect(self._obj(VAC_COLUMN_NAME), "Specs")
            point["vac_reboiler_temp"] = float(self._reflect(vac_specs2["R"], "SpecValue"))
        except Exception:
            point["vac_reboiler_temp"] = 360.0

        vac_temp_map = {
            "vac_diesel_draw_temp": PRODUCT_STREAMS["Vacuum_Diesel"],
            "vgo_draw_temp":        PRODUCT_STREAMS["Vacuum_Gas_Oil"],
        }
        for key, stream in vac_temp_map.items():
            try:
                point[key] = self.get_property(stream, "PROP_MS_0") - 273.15
            except Exception:
                point[key] = 0.0

        # --- Pressure parameters (kPa) ---
        try:
            point["atmos_top_pressure"] = self.get_property(ATMOS_COLUMN_NAME, "PROP_DC_0") / 1000
        except Exception:
            point["atmos_top_pressure"] = 101.0

        try:
            atop = self.get_property(ATMOS_COLUMN_NAME, "PROP_DC_0")
            abot = self.get_property(ATMOS_COLUMN_NAME, "PROP_DC_1")
            point["atmos_dp"] = (abot - atop) / 1000
        except Exception:
            point["atmos_dp"] = 15.0

        try:
            point["vac_top_pressure"] = self.get_property(VAC_COLUMN_NAME, "PROP_DC_0") / 1000
        except Exception:
            point["vac_top_pressure"] = 8.0

        try:
            vtop = self.get_property(VAC_COLUMN_NAME, "PROP_DC_0")
            vbot = self.get_property(VAC_COLUMN_NAME, "PROP_DC_1")
            point["vac_dp"] = (vbot - vtop) / 1000
        except Exception:
            point["vac_dp"] = 7.0

        logger.info(f"Current operating point read from simulation: {point}")
        return point

    def apply_action(self, action: dict[str, float]) -> None:
        """
        Apply incremental delta actions to the simulation.

        Each value in `action` is a per-step *delta* in physical units:
          - temperatures (draw temps, reboiler setpoints): °C
          - reflux ratios: dimensionless

        The current DWSIM value is read first, the delta is added, the result
        is clamped to _ACTION_HARD_LIMITS, and the new value is written back.
        Small incremental changes keep the column solver in its convergence
        basin — preventing the "max iterations" failures caused by large jumps.
        """
        _mapping: dict[str, tuple[str, str, bool]] = {
            # obj_name, prop_id, is_temp_stream
            "reflux_ratio":         (ATMOS_COLUMN_NAME,                    "__REFLUX__",   False),
            "hn_draw_temp":         (PRODUCT_STREAMS["Heavy_Naphtha"],     "PROP_MS_0",    True),
            "sko_draw_temp":        (PRODUCT_STREAMS["SKO"],               "PROP_MS_0",    True),
            "ld_draw_temp":         (PRODUCT_STREAMS["Light_Gas_Oil"],     "PROP_MS_0",    True),
            "hd_draw_temp":         (PRODUCT_STREAMS["Heavy_Gas_Oil"],     "PROP_MS_0",    True),
            "atmos_reboiler_temp":  (ATMOS_COLUMN_NAME,                    "__REBOILER__", False),
            "nsu_reflux_ratio":     (NSU_COLUMN_NAME,                      "__REFLUX__",   False),
            "nsu_reboiler_temp":    (NSU_COLUMN_NAME,                      "__REBOILER__", False),
            "vac_reflux_ratio":     (VAC_COLUMN_NAME,                      "__REFLUX__",   False),
            "vac_reboiler_temp":    (VAC_COLUMN_NAME,                      "__REBOILER__", False),
            "vac_diesel_draw_temp": (PRODUCT_STREAMS["Vacuum_Diesel"],     "PROP_MS_0",    True),
            "vgo_draw_temp":        (PRODUCT_STREAMS["Vacuum_Gas_Oil"],    "PROP_MS_0",    True),
        }
        # Pressure actions added to mapping
        _mapping["atmos_top_pressure"] = (ATMOS_COLUMN_NAME, "__TOP_PRESSURE__", False)
        _mapping["atmos_dp"]           = (ATMOS_COLUMN_NAME, "__COLUMN_DP__",    False)
        _mapping["vac_top_pressure"]   = (VAC_COLUMN_NAME,   "__TOP_PRESSURE__", False)
        _mapping["vac_dp"]             = (VAC_COLUMN_NAME,   "__COLUMN_DP__",    False)

        for key, delta in action.items():
            if key not in _mapping:
                continue
            obj_name, prop_id, is_temp_stream = _mapping[key]
            lo, hi = _ACTION_HARD_LIMITS.get(key, (-1e9, 1e9))

            if prop_id == "__TOP_PRESSURE__":
                # Condenser pressure: PROP_DC_0, stored in Pa, delta in kPa
                try:
                    current_pa = self.get_property(obj_name, "PROP_DC_0")
                    current_kpa = current_pa / 1000.0
                    new_kpa = max(lo, min(hi, current_kpa + float(delta)))
                    self.set_property(obj_name, "PROP_DC_0", new_kpa * 1000.0)
                    logger.debug(f"top_pressure {obj_name}: {current_kpa:.1f} + {delta:.2f} = {new_kpa:.1f} kPa")
                except Exception as exc:
                    logger.error(f"Failed to update top pressure for {obj_name}: {exc}")

            elif prop_id == "__COLUMN_DP__":
                # Pressure drop: adjust PROP_DC_1 (reboiler pressure)
                # DP = reboiler_pressure - condenser_pressure
                try:
                    top_pa = self.get_property(obj_name, "PROP_DC_0")
                    bot_pa = self.get_property(obj_name, "PROP_DC_1")
                    current_dp_kpa = (bot_pa - top_pa) / 1000.0
                    new_dp_kpa = max(lo, min(hi, current_dp_kpa + float(delta)))
                    new_bot_pa = top_pa + new_dp_kpa * 1000.0
                    self.set_property(obj_name, "PROP_DC_1", new_bot_pa)
                    logger.debug(f"DP {obj_name}: {current_dp_kpa:.1f} + {delta:.2f} = {new_dp_kpa:.1f} kPa")
                except Exception as exc:
                    logger.error(f"Failed to update DP for {obj_name}: {exc}")

            elif prop_id == "__REFLUX__":
                try:
                    col_obj = self._obj(obj_name)
                    specs   = self._reflect(col_obj, "Specs")
                    spec_c  = specs["C"]
                    current = float(self._reflect(spec_c, "SpecValue"))
                    new_val = max(lo, min(hi, current + float(delta)))
                    sv_prop = spec_c.GetType().GetProperty("SpecValue")
                    sv_prop.SetValue(spec_c, new_val, None)
                    logger.debug(f"reflux {obj_name}: {current:.3f} + {delta:.3f} = {new_val:.3f}")
                except Exception as exc:
                    logger.error(f"Failed to update reflux for {obj_name}: {exc}")

            elif prop_id == "__REBOILER__":
                try:
                    col_obj = self._obj(obj_name)
                    specs   = self._reflect(col_obj, "Specs")
                    spec_r  = specs["R"]
                    current = float(self._reflect(spec_r, "SpecValue"))
                    new_val = max(lo, min(hi, current + float(delta)))
                    sv_prop = spec_r.GetType().GetProperty("SpecValue")
                    sv_prop.SetValue(spec_r, new_val, None)
                    logger.debug(f"reboiler {obj_name}: {current:.1f} + {delta:.2f} = {new_val:.1f} °C")
                except Exception as exc:
                    logger.error(f"Failed to update reboiler for {obj_name}: {exc}")

            else:
                # Side-draw temperature: PROP_MS_0 stored in K, delta in °C (same magnitude)
                try:
                    current_k = self.get_property(obj_name, prop_id)
                    current_c = current_k - 273.15
                    new_c     = max(lo, min(hi, current_c + float(delta)))
                    self.set_property(obj_name, prop_id, new_c + 273.15)
                    logger.debug(f"{key} ({obj_name}): {current_c:.1f} + {delta:.2f} = {new_c:.1f} °C")
                except Exception as exc:
                    logger.error(f"Failed to update {key} ({obj_name}): {exc}")

    def apply_disturbance(self, disturbance: dict[str, float]) -> None:
        """Apply feed disturbances before solving."""
        try:
            if disturbance.get("feed_temperature_delta", 0) != 0:
                current = self.get_property(FEED_STREAM, "PROP_MS_0")
                self.set_property(
                    FEED_STREAM, "PROP_MS_0",
                    current + disturbance["feed_temperature_delta"],
                )

            if disturbance.get("feed_pressure_delta", 0) != 0:
                current = self.get_property(FEED_STREAM, "PROP_MS_1")
                self.set_property(
                    FEED_STREAM, "PROP_MS_1",
                    current + disturbance["feed_pressure_delta"] * 1000,  # kPa → Pa
                )

            if disturbance.get("feed_flow_delta", 0) != 0:
                current = self.get_property(FEED_STREAM, "PROP_MS_2")
                pct = disturbance["feed_flow_delta"] / 100.0
                self.set_property(FEED_STREAM, "PROP_MS_2", current * (1 + pct))
        except Exception as exc:
            logger.error(f"Error applying disturbance: {exc}")
