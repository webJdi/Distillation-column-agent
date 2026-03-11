import { useState, useEffect } from "react";
import {
  Flame,
  Thermometer,
  Gauge,
  Droplets,
  Beaker,
  ArrowDownUp,
  TrendingDown,
  TrendingUp,
  RotateCcw,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend,
} from "recharts";
import toast from "react-hot-toast";
import { analyzeDisturbance, disturbancePresets } from "../api";

const PRODUCT_COLORS = {
  Uncondensed_Gas: "#f59e0b",
  Heavy_Naphtha: "#8b5cf6",
  SKO: "#06b6d4",
  Light_Gas_Oil: "#22c55e",
  Heavy_Gas_Oil: "#14b8a6",
  StabOffGas: "#fb923c",
  LPG: "#3b82f6",
  SRN: "#60a5fa",
  Offgas: "#fbbf24",
  Vacuum_Diesel: "#ef4444",
  Vacuum_Gas_Oil: "#a855f7",
  Hotwell_Oil: "#64748b",
  Vac_residue: "#78350f",
};

function Slider({ label, icon: Icon, value, onChange, min, max, step, unit, color = "blue" }) {
  const pct = ((value - min) / (max - min)) * 100;
  const colors = {
    blue: "accent-blue-500",
    red: "accent-red-500",
    green: "accent-green-500",
    amber: "accent-amber-500",
    purple: "accent-purple-500",
  };

  return (
    <div className="glass-card p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Icon size={16} className="text-gray-400" />
          <span className="text-sm text-gray-300">{label}</span>
        </div>
        <span className="text-sm font-mono font-bold text-white">
          {value > 0 ? "+" : ""}
          {value}
          <span className="text-xs text-gray-500 ml-0.5">{unit}</span>
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={`w-full h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer ${colors[color]}`}
      />
      <div className="flex justify-between text-[10px] text-gray-600 mt-1">
        <span>{min}{unit}</span>
        <span>0</span>
        <span>+{max}{unit}</span>
      </div>
    </div>
  );
}

export default function DisturbancePage() {
  const [disturbance, setDisturbance] = useState({
    feed_temperature_delta: 0,
    feed_pressure_delta: 0,
    feed_flow_delta: 0,
    feed_api_gravity_delta: 0,
  });
  const [presets, setPresets] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    disturbancePresets()
      .then((r) => setPresets(r.data || []))
      .catch(() => {});
  }, []);

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const r = await analyzeDisturbance(disturbance);
      setResult(r.data);
      toast.success("Disturbance analysis complete");
    } catch (err) {
      toast.error("Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const handlePreset = (preset) => {
    setDisturbance(preset.disturbance);
    toast(`Applied: ${preset.name}`, { icon: "⚡" });
  };

  const handleReset = () => {
    setDisturbance({
      feed_temperature_delta: 0,
      feed_pressure_delta: 0,
      feed_flow_delta: 0,
      feed_api_gravity_delta: 0,
    });
    setResult(null);
  };

  // Build comparison chart data
  const comparisonData = result
    ? Object.entries(result.product_impact || {}).map(([name, impact]) => ({
        name,
        baseline: impact.baseline_flow,
        disturbed: impact.disturbed_flow,
        change: impact.change_percent,
      }))
    : [];

  return (
    <div className="space-y-6 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Feed Disturbances</h2>
          <p className="text-sm text-gray-500 mt-1">
            Introduce disturbances and see how the RL agent responds
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-800 text-gray-400 hover:text-white transition"
          >
            <RotateCcw size={16} /> Reset
          </button>
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-medium transition disabled:opacity-50"
          >
            <Flame size={16} />
            {loading ? "Analyzing…" : "Analyze Impact"}
          </button>
        </div>
      </div>

      {/* Preset quick buttons */}
      <div className="glass-card p-4">
        <p className="text-xs text-gray-500 mb-3">Quick Presets</p>
        <div className="flex flex-wrap gap-2">
          {presets.map((p) => (
            <button
              key={p.name}
              onClick={() => handlePreset(p)}
              className="px-3 py-1.5 rounded-lg bg-gray-800 text-xs text-gray-300 hover:bg-gray-700 hover:text-white border border-gray-700 transition"
              title={p.description}
            >
              {p.name}
            </button>
          ))}
        </div>
      </div>

      {/* Disturbance sliders */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Slider
          label="Feed Temperature"
          icon={Thermometer}
          value={disturbance.feed_temperature_delta}
          onChange={(v) =>
            setDisturbance((d) => ({ ...d, feed_temperature_delta: v }))
          }
          min={-50}
          max={50}
          step={1}
          unit="°C"
          color="red"
        />
        <Slider
          label="Feed Pressure"
          icon={Gauge}
          value={disturbance.feed_pressure_delta}
          onChange={(v) =>
            setDisturbance((d) => ({ ...d, feed_pressure_delta: v }))
          }
          min={-50}
          max={50}
          step={1}
          unit="kPa"
          color="blue"
        />
        <Slider
          label="Feed Flow Rate"
          icon={Droplets}
          value={disturbance.feed_flow_delta}
          onChange={(v) =>
            setDisturbance((d) => ({ ...d, feed_flow_delta: v }))
          }
          min={-30}
          max={30}
          step={1}
          unit="%"
          color="green"
        />
        <Slider
          label="API Gravity"
          icon={Beaker}
          value={disturbance.feed_api_gravity_delta}
          onChange={(v) =>
            setDisturbance((d) => ({ ...d, feed_api_gravity_delta: v }))
          }
          min={-10}
          max={10}
          step={0.5}
          unit=""
          color="purple"
        />
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-6 animate-fade-in">
          {/* Revenue impact summary */}
          <div className="glass-card p-6 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-xs text-gray-500">Baseline Revenue</p>
              <p className="text-xl font-bold text-white">
                ${result.revenue_impact?.baseline?.toFixed(0)}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-gray-500">Disturbed Revenue</p>
              <p className="text-xl font-bold text-white">
                ${result.revenue_impact?.disturbed?.toFixed(0)}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-gray-500">Change</p>
              <p
                className={`text-xl font-bold flex items-center justify-center gap-1 ${
                  result.revenue_impact?.change >= 0
                    ? "text-green-400"
                    : "text-red-400"
                }`}
              >
                {result.revenue_impact?.change >= 0 ? (
                  <TrendingUp size={18} />
                ) : (
                  <TrendingDown size={18} />
                )}
                ${Math.abs(result.revenue_impact?.change || 0).toFixed(0)}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-gray-500">Change %</p>
              <p
                className={`text-xl font-bold ${
                  result.revenue_impact?.change_percent >= 0
                    ? "text-green-400"
                    : "text-red-400"
                }`}
              >
                {result.revenue_impact?.change_percent >= 0 ? "+" : ""}
                {result.revenue_impact?.change_percent?.toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Comparison bar chart */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-4">
              Product Flow Comparison (Baseline vs Disturbed)
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="name"
                  stroke="#6b7280"
                  tick={{ fontSize: 12 }}
                />
                <YAxis
                  stroke="#6b7280"
                  tick={{ fontSize: 11 }}
                  label={{
                    value: "kg/h",
                    angle: -90,
                    position: "insideLeft",
                    fill: "#6b7280",
                    fontSize: 11,
                  }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1f2937",
                    border: "1px solid #374151",
                    borderRadius: 8,
                    color: "#fff",
                  }}
                />
                <Legend />
                <Bar
                  dataKey="baseline"
                  name="Baseline"
                  fill="#6b7280"
                  radius={[4, 4, 0, 0]}
                  opacity={0.6}
                />
                <Bar dataKey="disturbed" name="Disturbed" radius={[4, 4, 0, 0]}>
                  {comparisonData.map((entry) => (
                    <Cell
                      key={entry.name}
                      fill={PRODUCT_COLORS[entry.name] || "#6b7280"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Per-product impact table */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-4">
              Per-Product Impact Detail
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left py-2 text-gray-400 font-medium">Product</th>
                    <th className="text-right py-2 text-gray-400 font-medium">Baseline (kg/h)</th>
                    <th className="text-right py-2 text-gray-400 font-medium">Disturbed (kg/h)</th>
                    <th className="text-right py-2 text-gray-400 font-medium">Change %</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.product_impact || {}).map(
                    ([name, impact]) => (
                      <tr key={name} className="border-b border-gray-800/50">
                        <td className="py-2 text-white font-mono flex items-center gap-2">
                          <span
                            className="w-2.5 h-2.5 rounded-full"
                            style={{
                              backgroundColor:
                                PRODUCT_COLORS[name] || "#6b7280",
                            }}
                          />
                          {name}
                        </td>
                        <td className="py-2 text-right text-gray-300 font-mono">
                          {impact.baseline_flow?.toFixed(2)}
                        </td>
                        <td className="py-2 text-right text-gray-300 font-mono">
                          {impact.disturbed_flow?.toFixed(2)}
                        </td>
                        <td
                          className={`py-2 text-right font-mono font-medium ${
                            impact.change_percent >= 0
                              ? "text-green-400"
                              : "text-red-400"
                          }`}
                        >
                          {impact.change_percent >= 0 ? "+" : ""}
                          {impact.change_percent?.toFixed(1)}%
                        </td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Agent corrective action */}
          {result.agent_corrective_action && (
            <div className="glass-card p-6 ring-1 ring-green-500/20">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <ArrowDownUp size={18} className="text-green-400" />
                Agent&apos;s Corrective Action
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {Object.entries(result.agent_corrective_action).map(
                  ([key, val]) => (
                    <div
                      key={key}
                      className="bg-gray-800/50 rounded-lg px-3 py-2"
                    >
                      <p className="text-[11px] text-gray-500">
                        {key.replace(/_/g, " ")}
                      </p>
                      <p className="font-mono text-white font-bold">
                        {val.toFixed(2)}
                      </p>
                    </div>
                  )
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && (
        <div className="glass-card p-16 text-center">
          <Flame size={48} className="mx-auto text-gray-700 mb-4" />
          <p className="text-gray-500">
            Set disturbance parameters and click{" "}
            <strong>Analyze Impact</strong> to see how the column responds
          </p>
        </div>
      )}
    </div>
  );
}
