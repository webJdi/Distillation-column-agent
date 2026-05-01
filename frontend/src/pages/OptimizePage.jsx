import { useState, useEffect } from "react";
import { Activity, Zap, ArrowRight, Brain, Settings } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import toast from "react-hot-toast";
import { optimize, listCheckpoints, loadCheckpoint } from "../api";

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

export default function OptimizePage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [scenario, setScenario] = useState("default");
  const [checkpoints, setCheckpoints] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [numRuns, setNumRuns] = useState(50);

  useEffect(() => {
    listCheckpoints()
      .then((r) => {
        const cps = r.data || [];
        setCheckpoints(cps);
        if (cps.length > 0) setSelectedAgent(cps[0].path);
      })
      .catch(() => {});
  }, []);

  const handleOptimize = async () => {
    setLoading(true);
    try {
      // Load selected checkpoint first if available
      if (selectedAgent) {
        await loadCheckpoint(selectedAgent);
      }
      const r = await optimize({ scenario_name: scenario, num_runs: numRuns });
      setResult(r.data);
      toast.success(numRuns > 1 ? `Completed ${numRuns} optimization runs` : "Optimization complete");
    } catch (err) {
      toast.error(err.response?.data?.detail || "No trained model available");
    } finally {
      setLoading(false);
    }
  };

  const revenueData = result
    ? Object.entries(result.product_revenues || {}).map(([name, rev]) => ({
        name,
        revenue: rev,
      }))
    : [];

  return (
    <div className="space-y-6 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Optimize</h2>
          <p className="text-sm text-gray-500 mt-1">
            Run the trained RL agent to recommend optimal ADU + NSU + VDU column settings
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-white transition"
            title="Run settings"
          >
            <Settings size={18} />
          </button>
          <button
            onClick={handleOptimize}
            disabled={loading}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-green-600 hover:bg-green-700 text-white font-medium transition disabled:opacity-50"
          >
            <Zap size={16} />
            {loading ? "Optimizing…" : "Run Agent"}
          </button>
        </div>
      </div>

      {/* Settings panel */}
      {showSettings && (
        <div className="glass-card p-4 border border-blue-500/20 animate-fade-in">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Settings size={16} className="text-blue-400" />
              Run Configuration
            </h3>
            <button
              onClick={() => setShowSettings(false)}
              className="text-gray-500 hover:text-white"
            >
              ✕
            </button>
          </div>
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-gray-400 mb-2">
                Number of Optimization Runs <span className="text-gray-500">(default: 50, stochastic results averaged)</span>
              </label>
              <input
                type="number"
                min={1}
                max={200}
                step={1}
                value={numRuns}
                onChange={(e) => setNumRuns(Math.max(1, Number(e.target.value)))}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white font-mono outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-[11px] text-gray-600 mt-1">
                Results show average profit with ±std dev. Recommended settings from highest profit run.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Agent selector */}
      <div className="glass-card p-4 space-y-3">
        <div className="flex items-center gap-3">
          <Brain size={18} className="text-blue-400" />
          <span className="text-sm text-gray-400">Select Agent:</span>
          <select
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white outline-none focus:ring-2 focus:ring-blue-500"
            value={selectedAgent}
            onChange={(e) => setSelectedAgent(e.target.value)}
          >
            {checkpoints.length === 0 && (
              <option value="">No trained agents available</option>
            )}
            {checkpoints.map((cp) => {
              const algo = cp.name.split("_").find((s) => ["SAC", "PPO", "TD3"].includes(s)) || "?";
              const avgR = cp.metrics_summary?.avg_reward;
              const label = `${cp.name}  —  ${algo}${avgR != null ? `  |  Avg Reward: ${avgR.toFixed(4)}` : ""}`;
              return (
                <option key={cp.path} value={cp.path}>
                  {label}
                </option>
              );
            })}
          </select>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-400 ml-7">Price Scenario:</span>
          <input
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white w-48 outline-none focus:ring-2 focus:ring-blue-500"
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            placeholder="default"
          />
        </div>
      </div>

      {result && (
        <div className="space-y-6 animate-fade-in">
          {/* Profit header */}
          <div className="glass-card p-6 ring-1 ring-green-500/20 text-center">
            <p className="text-sm text-gray-400">
              {result.num_runs && result.num_runs > 1 
                ? `Average Hourly Profit (from ${result.num_runs} runs)` 
                : "Estimated Hourly Profit"}
            </p>
            <div className="mt-2">
              <p className="text-4xl font-bold text-green-400">
                ${result.avg_profit?.toFixed(2)}
                <span className="text-lg text-gray-500">/hr</span>
              </p>
              {result.std_profit !== undefined && result.std_profit > 0 && (
                <p className="text-sm text-gray-400 mt-1">
                  ±{result.std_profit?.toFixed(2)} std dev
                </p>
              )}
            </div>
            <div className="flex items-center justify-center gap-6 mt-3 text-xs text-gray-500">
              {result.avg_revenue > 0 && (
                <span>
                  Revenue: <b className="text-green-300">${result.avg_revenue?.toFixed(2)}/hr</b>
                </span>
              )}
              {result.avg_feed_cost > 0 && (
                <span>
                  Feed cost: <b className="text-red-300">${result.avg_feed_cost?.toFixed(2)}/hr</b>
                </span>
              )}
            </div>
          </div>

          {/* Recommended actions (from best run) */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Activity size={18} className="text-blue-400" />
              Recommended Column Settings
              {result.num_runs && result.num_runs > 1 && (
                <span className="text-sm text-gray-400 font-normal ml-auto">
                  (from best run: ${result.best_profit?.toFixed(2)}/hr)
                </span>
              )}
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {Object.entries(result.recommended_action || {}).map(
                ([key, value]) => (
                  <div key={key} className="bg-gray-800/50 rounded-lg p-4">
                    <p className="text-xs text-gray-500 mb-1">
                      {key
                        .replace(/_/g, " ")
                        .replace(/\b\w/g, (c) => c.toUpperCase())}
                    </p>
                    <p className="text-lg font-mono font-bold text-white">
                      {typeof value === "number" ? value.toFixed(2) : value}
                      <span className="text-xs text-gray-500 ml-1">
                        {key.includes("temp")
                          ? "°C"
                          : key.includes("steam")
                          ? "kg/h"
                          : ""}
                      </span>
                    </p>
                  </div>
                )
              )}
            </div>
          </div>

          {/* Revenue bar chart */}
          {revenueData.length > 0 && (
            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold text-white mb-4">
                Ave Product Revenue Breakdown
              </h3>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={revenueData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="name"
                    stroke="#6b7280"
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis
                    stroke="#6b7280"
                    tick={{ fontSize: 11 }}
                    tickFormatter={(v) => `$${v}`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1f2937",
                      border: "1px solid #374151",
                      borderRadius: 8,
                      color: "#fff",
                    }}
                    formatter={(v) => [`$${v.toFixed(2)}`, "Revenue"]}
                  />
                  <Bar dataKey="revenue" radius={[6, 6, 0, 0]}>
                    {revenueData.map((entry) => (
                      <Cell
                        key={entry.name}
                        fill={PRODUCT_COLORS[entry.name] || "#6b7280"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* D95% Quality */}
          {result.d95 && Object.keys(result.d95).length > 0 && (
            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold text-white mb-4">
                D95% Distillation Temperatures
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {Object.entries(result.d95).map(([name, temp]) => (
                  <div key={name} className="bg-gray-800/50 rounded-lg p-4 flex items-center gap-3">
                    <span
                      className="w-3 h-3 rounded-full flex-shrink-0"
                      style={{ backgroundColor: PRODUCT_COLORS[name] || "#6b7280" }}
                    />
                    <div className="min-w-0">
                      <p className="text-xs text-gray-500 truncate">{name.replace(/_/g, " ")}</p>
                      <p className="text-lg font-mono font-bold text-white">
                        {temp > 0 ? `${temp.toFixed(1)}` : "N/A"}
                        {temp > 0 && <span className="text-xs text-gray-500 ml-1">°C</span>}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Predicted state */}
          {result.predicted_state && (
            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold text-white mb-4">
                Predicted Column State
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                {Object.entries(result.predicted_state)
                  .filter(([, v]) => typeof v === "number")
                  .slice(0, 16)
                  .map(([key, value]) => (
                    <div
                      key={key}
                      className="bg-gray-800/50 rounded-lg px-3 py-2"
                    >
                      <p className="text-[11px] text-gray-500 truncate">
                        {key.replace(/_/g, " ")}
                      </p>
                      <p className="font-mono text-white">
                        {value.toFixed(2)}
                      </p>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && (
        <div className="glass-card p-16 text-center">
          <Zap size={48} className="mx-auto text-gray-700 mb-4" />
          <p className="text-gray-500">
            Click <strong>Run Agent</strong> to get optimal column settings
          </p>
          <p className="text-xs text-gray-600 mt-2">
            Make sure a trained model checkpoint is loaded first
          </p>
        </div>
      )}
    </div>
  );
}
