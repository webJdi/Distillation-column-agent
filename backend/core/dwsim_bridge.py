"""
DWSIM Automation Bridge — wraps the DWSIM COM/.NET automation API
so the rest of the backend never touches CLR directly.

The model uses main_sim.dwxmz which contains two columns:
  1. Atmospheric Distillation Column — object name: "Atmos_Column"
  2. Vacuum Distillation Column     — object name: "Vacuum_Column"

Material streams (17 total, as enumerated from the flowsheet):
  Crude inlet  : Upper_Zakum  (raw crude feed)
  Atmos feed   : Crude_to_column  (preheated crude to Atmos_Column)
  Atmos steam  : Bottom_Stripping_Steam
  ADU products : Uncondensed_Gas, Unstab_Naphtha, Heavy_Naphtha, SKO,
                 Light_Gas_Oil, Heavy_Gas_Oil
  Atmos bottom : Reduced_Crude_Oil  (RCO → RCO Pump → Vac_Furnace → Vac_feed)
  Vac feed     : Vac_feed  (heated RCO into Vacuum_Column)
  VDU products : Vacuum_Diesel, Vacuum_Gas_Oil, Hotwell_Oil, Vac_residue
  VDU overhead : OffGas  (non-product overhead — not priced)

Note: The vacuum column has no material steam stream; it uses an energy
stream (E8) on the reboiler side only.

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


# ── CDU + VDU object-name mapping ──────────────────────────────────────────
# These match the main_sim.dwxmz flowsheet exactly.

# Atmospheric column
ATMOS_COLUMN_NAME  = "Atmos_Column"
FEED_STREAM        = "Crude_to_column"                  # preheated crude into Atmos_Column
FEED_INPUT         = "Upper_Zakum"                       # raw crude inlet (before furnace)
FURNACE            = "Heater"                            # atmospheric furnace
ATMOS_STEAM        = "Bottom_Stripping_Steam"

# Vacuum column
VAC_COLUMN_NAME    = "Vacuum_Column"                     # exact tag in main_sim.dwxmz
VAC_FEED_STREAM    = "Vac_feed"                          # heated RCO entering Vacuum_Column
VAC_RCO_STREAM     = "Reduced_Crude_Oil"                 # RCO from atmos bottom → pump
VAC_FURNACE        = "Vac_Furnace"                       # vacuum heater
# Note: the vacuum column has NO material steam stream (uses energy stream E8)
VAC_STEAM          = None                                # not present in main_sim.dwxmz

# Legacy aliases (kept for backward-compat with other modules)
COLUMN_NAME  = ATMOS_COLUMN_NAME
STEAM_STREAM = ATMOS_STEAM

# ── Product streams (ADU + VDU) ────────────────────────────────────────────
PRODUCT_STREAMS: dict[str, str] = {
    # --- Atmospheric products ---
    "Uncondensed_Gas": "Uncondensed_Gas",       # overhead off-gas
    "USN":             "Unstab_Naphtha",         # unstabilized naphtha (≈ naphtha + LPG)
    "HN":              "Heavy_Naphtha",
    "SKO":             "SKO",                    # jet fuel / kerosene
    "LD":              "Light_Gas_Oil",           # light diesel
    "HD":              "Heavy_Gas_Oil",            # heavy diesel
    # --- Vacuum products (stream names as they appear in main_sim.dwxmz) ---
    "Vac_Diesel":      "Vacuum_Diesel",          # VDU side-draw diesel
    "VGO":             "Vacuum_Gas_Oil",          # vacuum gas oil
    "Slop_Cut":        "Hotwell_Oil",             # hotwell sump / slop cut
    "Vac_Residue":     "Vac_residue",             # vacuum residue (bottom)
    # Note: "OffGas" is the VDU overhead non-product stream (not priced)
}

# ── DWSIM property identifiers ─────────────────────────────────────────────
# Material Streams:
#   PROP_MS_0  = Temperature (K)
#   PROP_MS_1  = Pressure    (Pa)
#   PROP_MS_3  = Molar Flow  (mol/s)
#   PROP_MS_4  = Mass Flow   (kg/s)
#   PROP_MS_27 = Volumetric Flow (m³/s, at T,P of stream)
#
# Distillation Column:
#   PROP_DC_0  = Condenser Pressure (Pa)
#   PROP_DC_1  = Reboiler Pressure  (Pa)
#   PROP_DC_2  = Reflux Ratio
#   PROP_DC_5  = Condenser Duty (W)
#   PROP_DC_6  = Reboiler Duty  (W)
#   PROP_DC_7  = Number of Stages
#   PROP_DC_8  = Steam Rate


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
                mass_flow = self.get_property(stream, "PROP_MS_4")
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
        Returns a flat dict of floats covering both ADU and VDU.
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
            feed_flow = self.get_property(FEED_STREAM, "PROP_MS_4") * 3600  # kg/h
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
            vac_bot_temp = self.get_property(PRODUCT_STREAMS["Vac_Residue"], "PROP_MS_0") - 273.15
        except Exception:
            vac_bot_temp = 0.0

        return {
            **{f"flow_{k}": v for k, v in flows.items()},
            **{f"temp_{k}": v for k, v in temps.items()},
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

    # ── High-level CDU writes ───────────────────────────────────────────

    def apply_action(self, action: dict[str, float]) -> None:
        """
        Apply an RL action dict to the simulation.
        Expected keys (ADU): reflux_ratio, usn_draw_temp, hn_draw_temp,
                              sko_draw_temp, ld_draw_temp, hd_draw_temp,
                              atmos_steam_rate
        Expected keys (VDU): vac_reflux_ratio, vac_diesel_draw_temp,
                              vgo_draw_temp, vac_steam_rate
        """
        mapping: dict[str, tuple[str, str, bool]] = {
            # --- Atmospheric column ---
            "reflux_ratio":         (ATMOS_COLUMN_NAME,  "PROP_DC_2", False),
            "usn_draw_temp":        (PRODUCT_STREAMS["USN"], "PROP_MS_0", True),
            "hn_draw_temp":         (PRODUCT_STREAMS["HN"],  "PROP_MS_0", True),
            "sko_draw_temp":        (PRODUCT_STREAMS["SKO"], "PROP_MS_0", True),
            "ld_draw_temp":         (PRODUCT_STREAMS["LD"],  "PROP_MS_0", True),
            "hd_draw_temp":         (PRODUCT_STREAMS["HD"],  "PROP_MS_0", True),
            "atmos_steam_rate":     (ATMOS_STEAM, "PROP_MS_4", False),
            # --- Vacuum column ---
            "vac_reflux_ratio":     (VAC_COLUMN_NAME,  "PROP_DC_2", False),
            "vac_diesel_draw_temp": (PRODUCT_STREAMS["Vac_Diesel"], "PROP_MS_0", True),
            "vgo_draw_temp":        (PRODUCT_STREAMS["VGO"], "PROP_MS_0", True),
            # Note: vac_steam_rate removed — Vacuum_Column uses energy stream E8, not steam
        }
        for key, (obj_name, prop_id, is_temp) in mapping.items():
            if key in action:
                value = action[key]
                if is_temp:
                    value += 273.15  # °C → K
                if "steam" in key:
                    value /= 3600  # kg/h → kg/s
                self.set_property(obj_name, prop_id, value)

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
                current = self.get_property(FEED_STREAM, "PROP_MS_4")
                pct = disturbance["feed_flow_delta"] / 100.0
                self.set_property(FEED_STREAM, "PROP_MS_4", current * (1 + pct))
        except Exception as exc:
            logger.error(f"Error applying disturbance: {exc}")
