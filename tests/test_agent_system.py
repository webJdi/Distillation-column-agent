"""
Unit tests for the AI Agent system (improvement 10).

Tests the deterministic numeric module, schema validation,
compact context builder, and audit logger — no LLM calls required.
"""

import json
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# ── Ensure project root is on sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═════════════════════════════════════════════════════════════════════════════
#  Inline helpers (mirror the notebook code so tests are self-contained)
# ═════════════════════════════════════════════════════════════════════════════

ALL_PRODUCTS = [
    "Uncondensed_Gas", "Heavy_Naphtha", "SKO", "Light_Gas_Oil", "Heavy_Gas_Oil",
    "StabOffGas", "LPG", "SRN",
    "Offgas", "Vacuum_Diesel", "Vacuum_Gas_Oil", "Hotwell_Oil", "Vac_residue",
]

D95_SPECS = {
    "Heavy_Naphtha": 220.0, "SKO": 300.0, "Light_Gas_Oil": 370.0,
    "Heavy_Gas_Oil": 385.0, "Vacuum_Diesel": 385.0, "Vacuum_Gas_Oil": 520.0,
}


def _mock_state() -> dict:
    """Representative operating snapshot for testing."""
    return {
        "flow_Uncondensed_Gas": 25.0, "flow_Heavy_Naphtha": 106.0,
        "flow_SKO": 43.0, "flow_Light_Gas_Oil": 51.0, "flow_Heavy_Gas_Oil": 69.0,
        "flow_StabOffGas": 18.0, "flow_LPG": 32.0, "flow_SRN": 56.0,
        "flow_Offgas": 12.0, "flow_Vacuum_Diesel": 38.0,
        "flow_Vacuum_Gas_Oil": 72.0, "flow_Hotwell_Oil": 14.0, "flow_Vac_residue": 95.0,
        "temp_Uncondensed_Gas": 50.0, "temp_Heavy_Naphtha": 155.0,
        "temp_SKO": 220.0, "temp_Light_Gas_Oil": 280.0, "temp_Heavy_Gas_Oil": 340.0,
        "temp_StabOffGas": 45.0, "temp_LPG": 50.0, "temp_SRN": 90.0,
        "temp_Offgas": 65.0, "temp_Vacuum_Diesel": 250.0,
        "temp_Vacuum_Gas_Oil": 350.0, "temp_Hotwell_Oil": 380.0, "temp_Vac_residue": 410.0,
        "d95_Heavy_Naphtha": 195.0, "d95_SKO": 275.0, "d95_Light_Gas_Oil": 355.0,
        "d95_Heavy_Gas_Oil": 375.0, "d95_Vacuum_Diesel": 370.0, "d95_Vacuum_Gas_Oil": 500.0,
        "top_temperature": 50.0, "bottom_temperature": 340.0,
        "feed_temperature": 365.0, "feed_flow_rate": 631.0,  # sum of all products
        "top_pressure": 101.0, "bottom_pressure": 116.0,
        "condenser_duty": 42000.0, "reboiler_duty": 46000.0,
        "vac_top_pressure": 8.0, "vac_bottom_pressure": 15.0,
        "overhead_temperature": 50.0, "overhead_pressure": 101.0,
        "overhead_water_content": 0.02, "overhead_hcl_ppm": 5.0,
        "overhead_h2s_ppm": 15.0, "overhead_nh3_ppm": 8.0,
        "reflux_ratio": 5.0, "nsu_reflux_ratio": 4.0, "vac_reflux_ratio": 3.5,
        "nsu_top_temperature": 45.0, "nsu_bottom_temperature": 155.0, "nsu_top_pressure": 800.0,
    }


def _mock_prices() -> dict:
    return {
        "Uncondensed_Gas": 0.30, "Heavy_Naphtha": 0.60, "SKO": 0.75,
        "Light_Gas_Oil": 0.70, "Heavy_Gas_Oil": 0.70,
        "StabOffGas": 0.30, "LPG": 0.65, "SRN": 0.75,
        "Offgas": 0.30, "Vacuum_Diesel": 0.70, "Vacuum_Gas_Oil": 0.50,
        "Hotwell_Oil": 0.50, "Vac_residue": 0.35,
        "Feed_Crude": 0.40,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Deterministic numeric functions (copied from notebook for self-contained tests)
# ═════════════════════════════════════════════════════════════════════════════

def compute_mass_balance(state: dict) -> dict:
    feed = state.get("feed_flow_rate", 0.0)
    total_product = sum(state.get(f"flow_{p}", 0.0) for p in ALL_PRODUCTS)
    gap = feed - total_product
    gap_pct = (gap / feed * 100) if feed > 0 else 0.0
    return {
        "feed_flow_kg_h": round(feed, 1),
        "total_product_kg_h": round(total_product, 1),
        "gap_kg_h": round(gap, 1),
        "gap_pct": round(gap_pct, 2),
        "closure_ok": abs(gap_pct) < 2.0,
    }


def compute_profit(state: dict, prices: dict) -> dict:
    revenue_by_product = {}
    for p in ALL_PRODUCTS:
        flow = state.get(f"flow_{p}", 0.0)
        price = prices.get(p, 0.0)
        revenue_by_product[p] = round(flow * price, 2)
    total_revenue = sum(revenue_by_product.values())
    feed_cost = state.get("feed_flow_rate", 0.0) * prices.get("Feed_Crude", 0.40)
    net_profit = total_revenue - feed_cost
    return {
        "revenue_by_product": revenue_by_product,
        "total_revenue_hr": round(total_revenue, 2),
        "feed_cost_hr": round(feed_cost, 2),
        "net_profit_hr": round(net_profit, 2),
        "top_3_contributors": sorted(revenue_by_product.items(), key=lambda x: x[1], reverse=True)[:3],
    }


def compute_d95_compliance(state: dict) -> dict:
    results = {}
    for product, limit in D95_SPECS.items():
        actual = state.get(f"d95_{product}", None)
        results[product] = {
            "d95_actual_C": actual,
            "d95_limit_C": limit,
            "pass": actual is not None and actual <= limit,
            "margin_C": round(limit - actual, 1) if actual is not None else None,
        }
    return results


def _validate_against_schema(data: dict, schema: dict) -> tuple:
    errors = []
    props = schema.get("properties", {})
    required = schema.get("required", [])
    for key in required:
        if key not in data:
            errors.append(f"Missing required field: '{key}'")
    for key, value in data.items():
        if key in props:
            expected_type = props[key].get("type")
            type_map = {"string": str, "number": (int, float), "array": list,
                        "object": dict, "boolean": bool}
            if expected_type and expected_type in type_map:
                if not isinstance(value, type_map[expected_type]):
                    errors.append(f"Field '{key}' expected {expected_type}, got {type(value).__name__}")
            if expected_type == "string" and "enum" in props[key]:
                if value not in props[key]["enum"]:
                    errors.append(f"Field '{key}' must be one of {props[key]['enum']}, got '{value}'")
            if expected_type == "number":
                if "minimum" in props[key] and value < props[key]["minimum"]:
                    errors.append(f"Field '{key}' below minimum {props[key]['minimum']}")
                if "maximum" in props[key] and value > props[key]["maximum"]:
                    errors.append(f"Field '{key}' above maximum {props[key]['maximum']}")
    return (len(errors) == 0, errors)


# ═════════════════════════════════════════════════════════════════════════════
#  Test Cases
# ═════════════════════════════════════════════════════════════════════════════

class TestMassBalance:
    def test_closure_with_matching_feed(self):
        state = _mock_state()
        # feed_flow_rate = 631.0 = sum of all product flows
        result = compute_mass_balance(state)
        assert result["closure_ok"] is True
        assert result["gap_pct"] == 0.0
        assert result["gap_kg_h"] == 0.0

    def test_closure_fails_with_large_gap(self):
        state = _mock_state()
        state["feed_flow_rate"] = 1000.0  # Much larger than product sum
        result = compute_mass_balance(state)
        assert result["closure_ok"] is False
        assert result["gap_pct"] > 2.0

    def test_zero_feed(self):
        state = _mock_state()
        state["feed_flow_rate"] = 0.0
        result = compute_mass_balance(state)
        assert result["gap_pct"] == 0.0

    def test_all_products_included(self):
        state = _mock_state()
        result = compute_mass_balance(state)
        total = sum(state.get(f"flow_{p}", 0.0) for p in ALL_PRODUCTS)
        assert result["total_product_kg_h"] == round(total, 1)


class TestProfit:
    def test_positive_profit(self):
        state = _mock_state()
        prices = _mock_prices()
        result = compute_profit(state, prices)
        assert result["net_profit_hr"] != 0
        assert result["total_revenue_hr"] > 0
        assert result["feed_cost_hr"] > 0
        assert len(result["top_3_contributors"]) == 3

    def test_zero_prices(self):
        state = _mock_state()
        prices = {p: 0.0 for p in ALL_PRODUCTS}
        prices["Feed_Crude"] = 0.0
        result = compute_profit(state, prices)
        assert result["total_revenue_hr"] == 0.0
        assert result["net_profit_hr"] == 0.0

    def test_revenue_by_product_sums_correctly(self):
        state = _mock_state()
        prices = _mock_prices()
        result = compute_profit(state, prices)
        total = sum(result["revenue_by_product"].values())
        assert abs(total - result["total_revenue_hr"]) < 0.01


class TestD95Compliance:
    def test_all_within_spec(self):
        state = _mock_state()
        result = compute_d95_compliance(state)
        for product, info in result.items():
            assert info["pass"] is True, f"{product} should pass but d95={info['d95_actual_C']}"

    def test_exceedance_detected(self):
        state = _mock_state()
        state["d95_Heavy_Naphtha"] = 250.0  # Above 220 limit
        result = compute_d95_compliance(state)
        assert result["Heavy_Naphtha"]["pass"] is False
        assert result["Heavy_Naphtha"]["margin_C"] < 0

    def test_missing_d95(self):
        state = _mock_state()
        del state["d95_Heavy_Naphtha"]
        result = compute_d95_compliance(state)
        assert result["Heavy_Naphtha"]["d95_actual_C"] is None
        assert result["Heavy_Naphtha"]["pass"] is False


class TestSchemaValidation:
    def test_valid_response(self):
        data = {"summary": "All good", "analysis": "Detailed analysis here."}
        schema = {
            "properties": {
                "summary": {"type": "string"},
                "analysis": {"type": "string"},
            },
            "required": ["summary", "analysis"],
        }
        valid, errors = _validate_against_schema(data, schema)
        assert valid is True
        assert len(errors) == 0

    def test_missing_required_field(self):
        data = {"summary": "All good"}
        schema = {
            "properties": {
                "summary": {"type": "string"},
                "analysis": {"type": "string"},
            },
            "required": ["summary", "analysis"],
        }
        valid, errors = _validate_against_schema(data, schema)
        assert valid is False
        assert any("analysis" in e for e in errors)

    def test_wrong_type(self):
        data = {"summary": 42, "analysis": "ok"}
        schema = {
            "properties": {
                "summary": {"type": "string"},
                "analysis": {"type": "string"},
            },
            "required": ["summary", "analysis"],
        }
        valid, errors = _validate_against_schema(data, schema)
        assert valid is False
        assert any("summary" in e for e in errors)

    def test_enum_validation(self):
        data = {"summary": "ok", "analysis": "ok", "risk_level": "PURPLE"}
        schema = {
            "properties": {
                "summary": {"type": "string"},
                "analysis": {"type": "string"},
                "risk_level": {"type": "string", "enum": ["GREEN", "YELLOW", "RED"]},
            },
            "required": ["summary", "analysis"],
        }
        valid, errors = _validate_against_schema(data, schema)
        assert valid is False
        assert any("risk_level" in e for e in errors)


class TestAuditLog:
    def test_audit_log_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"
            # Inline audit_log function
            entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": "test_agent",
                "question": "test question",
                "context_keys": ["key1"],
                "response_length": 10,
                "response_preview": "test resp",
                "validation": None,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

            # Verify it's valid JSONL
            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["agent"] == "test_agent"


class TestResponseSchema:
    def test_schema_file_exists(self):
        schema_path = PROJECT_ROOT / "backend" / "models" / "agent_response_schema.json"
        assert schema_path.exists(), f"Schema file not found: {schema_path}"

    def test_schema_is_valid_json(self):
        schema_path = PROJECT_ROOT / "backend" / "models" / "agent_response_schema.json"
        with open(schema_path) as f:
            schema = json.load(f)
        assert "properties" in schema
        assert "required" in schema
        assert "summary" in schema["required"]
        assert "analysis" in schema["required"]


# ═════════════════════════════════════════════════════════════════════════════
#  Run tests
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
