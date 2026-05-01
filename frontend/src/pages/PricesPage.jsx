import { useState, useEffect } from "react";
import { DollarSign, Save, Plus, List } from "lucide-react";
import toast from "react-hot-toast";
import { savePrices, getPrices, listScenarios, createScenario } from "../api";

// All prices in $/kg — consistent with backend reward calculation (flow kg/h × price $/kg = $/h)
const PRODUCTS = [
  { key: "Feed_Crude",      label: "Feed Crude",     desc: "Crude oil feed cost ($/kg)",        color: "#dc2626" },
  { key: "Uncondensed_Gas", label: "Uncond. Gas",    desc: "Uncondensed Gas, ADU overhead",     color: "#f59e0b" },
  { key: "Heavy_Naphtha",   label: "Heavy Naphtha",  desc: "Heavy Naphtha ($/kg)",              color: "#8b5cf6" },
  { key: "SKO",             label: "SKO",            desc: "Superior Kerosene / Jet Fuel",      color: "#06b6d4" },
  { key: "Light_Gas_Oil",   label: "Light Gas Oil",  desc: "Light Gas Oil ($/kg)",              color: "#22c55e" },
  { key: "Heavy_Gas_Oil",   label: "Heavy Gas Oil",  desc: "Heavy Gas Oil ($/kg)",              color: "#14b8a6" },
  { key: "StabOffGas",      label: "Stab OffGas",   desc: "Naphtha Stabilizer off-gas",        color: "#fb923c" },
  { key: "LPG",             label: "LPG",            desc: "Liquefied Petroleum Gas ($/kg)",    color: "#3b82f6" },
  { key: "SRN",             label: "SRN",            desc: "Straight-Run Naphtha ($/kg)",       color: "#60a5fa" },
  { key: "Offgas",          label: "Offgas",         desc: "Vacuum overhead gas ($/kg)",        color: "#fbbf24" },
  { key: "Vacuum_Diesel",   label: "Vac Diesel",    desc: "Vacuum Diesel ($/kg)",              color: "#ef4444" },
  { key: "Vacuum_Gas_Oil",  label: "VGO",            desc: "Vacuum Gas Oil ($/kg)",             color: "#a855f7" },
  { key: "Hotwell_Oil",     label: "Hotwell Oil",   desc: "Hotwell Oil ($/kg)",                color: "#64748b" },
  { key: "Vac_residue",     label: "Vac Residue",   desc: "Vacuum Residue ($/kg)",             color: "#78350f" },
];

// $/kg — matches backend DEFAULT_PRICES in rl_environment.py
const DEFAULT_PRICES = {
  Feed_Crude: 0.45, Uncondensed_Gas: 0.30, Heavy_Naphtha: 0.60, SKO: 0.75,
  Light_Gas_Oil: 0.70, Heavy_Gas_Oil: 0.55, StabOffGas: 0.20, LPG: 0.55,
  SRN: 0.65, Offgas: 0.15, Vacuum_Diesel: 0.52, Vacuum_Gas_Oil: 0.45,
  Hotwell_Oil: 0.30, Vac_residue: 0.25,
};

export default function PricesPage() {
  const [prices, setPrices] = useState({ ...DEFAULT_PRICES });
  const [scenarioName, setScenarioName] = useState("default");
  const [scenarios, setScenarios] = useState([]);
  const [newScenario, setNewScenario] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadScenarios();
    loadPrices("default");
  }, []);

  const loadScenarios = async () => {
    try {
      const r = await listScenarios();
      setScenarios(Array.isArray(r.data) ? r.data : []);
    } catch {
      /* backend offline */
    }
  };

  const loadPrices = async (name) => {
    try {
      const r = await getPrices(name);
      if (r.data?.prices) {
        // Merge loaded prices with defaults, using defaults for any 0 values or missing keys
        const merged = { ...DEFAULT_PRICES };
        for (const [key, value] of Object.entries(r.data.prices)) {
          merged[key] = (value && value > 0) ? value : DEFAULT_PRICES[key];
        }
        setPrices(merged);
        setScenarioName(name);
      }
    } catch {
      /* use defaults */
    }
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      await savePrices({ ...prices, scenario_name: scenarioName });
      toast.success(`Prices saved for "${scenarioName}"`);
      loadScenarios();
    } catch (err) {
      toast.error("Failed to save prices");
    } finally {
      setLoading(false);
    }
  };

  const handleCreateScenario = async () => {
    if (!newScenario.trim()) return;
    try {
      await createScenario({
        name: newScenario.trim(),
        prices: { ...prices, scenario_name: newScenario.trim() },
      });
      toast.success(`Scenario "${newScenario}" created`);
      setScenarioName(newScenario.trim());
      setNewScenario("");
      loadScenarios();
    } catch {
      toast.error("Failed to create scenario");
    }
  };

  // Sum of all product prices minus feed price
  const totalRevenue = Object.entries(prices)
    .filter(([k]) => k !== "Feed_Crude")
    .reduce((sum, [, v]) => sum + v, 0) - (prices.Feed_Crude || 0);

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-2xl font-bold text-white">Product Prices</h2>
        <p className="text-sm text-gray-500 mt-1">
          Set market prices for each ADU + NSU + VDU product stream
        </p>
      </div>

      {/* Scenario selector */}
      <div className="glass-card p-5 flex flex-wrap items-center gap-4">
        <List size={18} className="text-gray-400" />
        <select
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-blue-500 outline-none"
          value={scenarioName}
          onChange={(e) => loadPrices(e.target.value)}
        >
          <option value="default">Default</option>
          {scenarios.map((s) => (
            <option key={s.scenario_name} value={s.scenario_name}>
              {s.scenario_name}
            </option>
          ))}
        </select>

        <div className="flex items-center gap-2 ml-auto">
          <input
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-500 w-40 focus:ring-2 focus:ring-blue-500 outline-none"
            placeholder="New scenario…"
            value={newScenario}
            onChange={(e) => setNewScenario(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreateScenario()}
          />
          <button
            onClick={handleCreateScenario}
            className="p-2 rounded-lg bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700 transition"
          >
            <Plus size={18} />
          </button>
        </div>
      </div>

      {/* Price inputs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {PRODUCTS.map(({ key, label, desc, color }) => (
          <div key={key} className={`glass-card p-5 animate-fade-in ${key === "Feed_Crude" ? "ring-1 ring-red-500/20" : ""}`}>
            <div className="flex items-center gap-3 mb-3">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color }}
              />
              <div>
                <p className="text-sm font-semibold text-white">{label}</p>
                <p className="text-[11px] text-gray-500">{desc}</p>
              </div>
            </div>
            <div className="relative">
              <DollarSign
                size={16}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500"
              />
              <input
                type="number"
                min={0}
                step={0.01}
                value={prices[key] || DEFAULT_PRICES[key] || 0}
                onChange={(e) =>
                  setPrices((p) => ({ ...p, [key]: Number(e.target.value) }))
                }
                className="w-full bg-gray-800 border border-gray-700 rounded-lg pl-9 pr-16 py-2.5 text-white text-right font-mono focus:ring-2 focus:ring-blue-500 outline-none"
              />
              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-500">
                $/kg
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Summary + Save */}
      <div className="glass-card p-5 flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-400">Sum of all prices</p>
          <p className="text-xl font-bold text-white">
            ${totalRevenue.toFixed(2)}
            <span className="text-sm text-gray-500 ml-1">$/kg (total)</span>
          </p>
        </div>
        <button
          onClick={handleSave}
          disabled={loading}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium transition disabled:opacity-50"
        >
          <Save size={16} />
          {loading ? "Saving…" : "Save Prices"}
        </button>
      </div>
    </div>
  );
}
