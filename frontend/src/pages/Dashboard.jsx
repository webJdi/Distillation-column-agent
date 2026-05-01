import { useEffect, useRef, useState } from "react";
import {
  Gauge,
  TrendingUp,
  Droplets,
  Zap,
  Brain,
  Activity,
  BarChart3,
  Target,
} from "lucide-react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell,
} from "recharts";
import { trainingStatus, getProductFlows, getSimState, getD95, getOptimizationScope, getKpiStats, resetKpiStats } from "../api";
import { useTrainingWebSocket } from "../useWebSocket";

const C = {
  blue: "#3b82f6",
  green: "#22c55e",
  purple: "#a855f7",
  orange: "#f97316",
  red: "#ef4444",
  cyan: "#06b6d4",
  yellow: "#eab308",
  pink: "#ec4899",
  slate: "#64748b",
};

const ACTION_COLORS = [C.blue, C.green, C.purple, C.orange, C.red, C.cyan, C.yellow, C.pink, C.slate, "#78350f"];

const ttStyle = {
  backgroundColor: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 8,
  color: "#e2e8f0",
  fontSize: 12,
};

function ChartCard({ title, icon: Icon, children }) {
  return (
    <div className="glass-card p-4">
      <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
        {Icon && <Icon size={15} className="text-gray-500" />}
        {title}
      </h4>
      {children}
    </div>
  );
}

/* ── Stat card ────────────────────────────────────────────────── */
function Stat({ icon: Icon, label, value, unit, color = "blue" }) {
  const ring = {
    blue: "ring-blue-500/30",
    green: "ring-green-500/30",
    amber: "ring-amber-500/30",
    red: "ring-red-500/30",
    purple: "ring-purple-500/30",
  }[color];
  const text = {
    blue: "text-blue-400",
    green: "text-green-400",
    amber: "text-amber-400",
    red: "text-red-400",
    purple: "text-purple-400",
  }[color];

  return (
    <div className={`glass-card p-5 ring-1 ${ring} animate-fade-in`}>
      <div className="flex items-center gap-3 mb-3">
        <div className={`p-2 rounded-lg ${text}`}>
          <Icon size={20} />
          <p className="text-5xl font-bold text-white">
            {value}
            {unit && <span className="text-sm text-gray-500 ml-1">{unit}</span>}
          </p>
          <span className="text-sm text-gray-400">{label}</span>
        </div>
        
      </div>
      
    </div>
  );
}

/* ── Product colour palette ───────────────────────────────────── */
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

const ACTION_LABELS = {
  reflux_ratio:         { label: "Reflux Ratio",         unit: "—"   },
  hn_draw_temp:         { label: "HN Draw Temp",         unit: "°C"  },
  sko_draw_temp:        { label: "SKO Draw Temp",        unit: "°C"  },
  ld_draw_temp:         { label: "LD Draw Temp",         unit: "°C"  },
  hd_draw_temp:         { label: "HD Draw Temp",         unit: "°C"  },
  atmos_reboiler_temp:  { label: "Atmos Reboiler",       unit: "°C"  },
  atmos_top_pressure:   { label: "Atmos Top Press.",     unit: "kPa" },
  atmos_dp:             { label: "Atmos ΔP",             unit: "kPa" },
  nsu_reflux_ratio:     { label: "NSU Reflux Ratio",     unit: "—"   },
  nsu_reboiler_temp:    { label: "NSU Reboiler",         unit: "°C"  },
  vac_reflux_ratio:     { label: "Vac Reflux Ratio",     unit: "—"   },
  vac_reboiler_temp:    { label: "Vac Reboiler",         unit: "°C"  },
  vac_diesel_draw_temp: { label: "Vac Diesel Draw",      unit: "°C"  },
  vgo_draw_temp:        { label: "VGO Draw Temp",        unit: "°C"  },
  vac_top_pressure:     { label: "Vac Top Press.",       unit: "kPa" },
  vac_dp:               { label: "Vac ΔP",               unit: "kPa" },
};

// Short display labels for X-axis
const PRODUCT_LABELS = {
  Uncondensed_Gas: "UnCond",
  Heavy_Naphtha: "Hvy Naph",
  SKO: "SKO",
  Light_Gas_Oil: "LGO",
  Heavy_Gas_Oil: "HGO",
  StabOffGas: "StabGas",
  LPG: "LPG",
  SRN: "SRN",
  Offgas: "Offgas",
  Vacuum_Diesel: "Vac Dsl",
  Vacuum_Gas_Oil: "VGO",
  Hotwell_Oil: "Hotwl",
  Vac_residue: "Vac Res",
};

// placeholder — kept so nothing else breaks if referenced elsewhere
function ProductBar({ name, flow, maxFlow = 1000 }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-gray-400">{name}</span>
      <span className="text-xs font-mono text-gray-300">{flow.toFixed(1)} kg/h</span>
    </div>
  );
}

/* ── Dashboard page ───────────────────────────────────────────── */
export default function Dashboard() {
  const { progress, connected } = useTrainingWebSocket();
  const [status, setStatus] = useState(null);
  const [throughput, setThroughput] = useState(null);
  const [d95, setD95] = useState({});
  const [scope, setScope] = useState(null);
  const [kpiStats, setKpiStats] = useState(null);

  // Persist best optimized profit and recommendations across no-improvement polls
  const [bestOptimizedProfit, setBestOptimizedProfit] = useState(null);
  const [lastRecommendations, setLastRecommendations] = useState(null);

  // Training analytics history
  const [rewardHistory, setRewardHistory] = useState([]);
  const [qHistory, setQHistory] = useState([]);
  const [gradHistory, setGradHistory] = useState([]);
  const [actionDist, setActionDist] = useState(null);
  const lastStep = useRef(-1);

  // Product flows for display (ADU + NSU + VDU)
  const [flows, setFlows] = useState({
    Uncondensed_Gas: 0,
    Heavy_Naphtha: 0,
    SKO: 0,
    Light_Gas_Oil: 0,
    Heavy_Gas_Oil: 0,
    StabOffGas: 0,
    LPG: 0,
    SRN: 0,
    Offgas: 0,
    Vacuum_Diesel: 0,
    Vacuum_Gas_Oil: 0,
    Hotwell_Oil: 0,
    Vac_residue: 0,
  });

  useEffect(() => {
    // Initial load
    trainingStatus()
      .then((r) => setStatus(r.data))
      .catch(() => {});
    getProductFlows()
      .then((r) => {
        if (r.data?.flows) setFlows(r.data.flows);
      })
      .catch(() => {});
    getSimState()
      .then((r) => {
        if (r.data?.state?.feed_flow_rate != null)
          setThroughput(r.data.state.feed_flow_rate);
      })
      .catch(() => {});
    getD95()
      .then((r) => {
        if (r.data?.d95) setD95(r.data.d95);
      })
      .catch(() => {});
    getOptimizationScope()
      .then((r) => { if (r.data?.available) setScope(r.data); })
      .catch(() => {});
    getKpiStats()
      .then((r) => setKpiStats(r.data || null))
      .catch(() => {});

    // Live polling interval for near real-time dashboard updates
    const interval = setInterval(() => {
      getProductFlows()
        .then((r) => { if (r.data?.flows) setFlows(r.data.flows); })
        .catch(() => {});
      getD95()
        .then((r) => { if (r.data?.d95) setD95(r.data.d95); })
        .catch(() => {});
      getOptimizationScope()
        .then((r) => { if (r.data?.available) setScope(r.data); })
        .catch(() => {});
      getKpiStats()
        .then((r) => setKpiStats(r.data || null))
        .catch(() => {});
      getSimState()
        .then((r) => {
          if (r.data?.state?.feed_flow_rate != null)
            setThroughput(r.data.state.feed_flow_rate);
        })
        .catch(() => {});
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Update best profit + recommendations whenever scope returns a genuine improvement
  useEffect(() => {
    if (!scope || scope.no_improvement) return;
    if (scope.optimized_profit != null) {
      setBestOptimizedProfit((prev) =>
        prev === null || scope.optimized_profit > prev ? scope.optimized_profit : prev
      );
    }
    if (scope.recommendations && Object.keys(scope.recommendations).length > 0) {
      setLastRecommendations(scope.recommendations);
    }
  }, [scope]);

  // Accumulate training analytics from WebSocket
  useEffect(() => {
    if (!progress || progress.status !== "training") return;
    const step = progress.current_step || 0;
    if (step === lastStep.current) return;
    lastStep.current = step;

    if (progress.episode > 0) {
      setRewardHistory((h) => {
        const next = [...h, { step, episode: progress.episode, avg: progress.avg_reward, best: progress.best_reward }];
        return next.length > 600 ? next.slice(-600) : next;
      });
    }
    if (progress.mean_q_value !== undefined || progress.entropy !== undefined) {
      setQHistory((h) => {
        const next = [...h, { step, q: progress.mean_q_value, entropy: progress.entropy, alpha: progress.ent_coef }];
        return next.length > 600 ? next.slice(-600) : next;
      });
    }
    if (progress.actor_grad_norm !== undefined || progress.critic_grad_norm !== undefined) {
      setGradHistory((h) => {
        const next = [...h, { step, actor: progress.actor_grad_norm, critic: progress.critic_grad_norm }];
        return next.length > 600 ? next.slice(-600) : next;
      });
    }
    if (progress.action_distribution) setActionDist(progress.action_distribution);
  }, [progress]);

  const agentStatus = progress?.status || status?.status || "idle";
  const avgReward = progress?.avg_reward?.toFixed(2) || "—";
  const episode = progress?.episode || 0;
  const profit = progress?.profit?.toFixed(0) || "—";

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Dashboard</h2>
          <p className="text-sm text-gray-500 mt-1">
            CDU RTO Overview
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`w-2.5 h-2.5 rounded-full ${
              connected ? "bg-green-500 pulse-glow" : "bg-red-500"
            }`}
          />
          <span className="text-xs text-gray-400">
            {connected ? "Live" : "Offline"}
          </span>
          <button
            type="button"
            onClick={async () => {
              try {
                const res = await resetKpiStats();
                setKpiStats(res.data || null);
                setBestOptimizedProfit(null);
              } catch (error) {
                console.error("Failed to reset KPI stats", error);
              }
            }}
            className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-white transition hover:bg-white/10"
          >
            Reset KPIs
          </button>
        </div>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Stat
          icon={Droplets}
          label="Avg Distillate Yield Improvement"
          value={kpiStats?.summary?.avg_distillate_yield_improvement?.toFixed(1) || "—"}
          unit="kg/h"
          color="blue"
        />
        <Stat
          icon={Zap}
          label="Cumulative Energy Savings"
          value={(kpiStats?.summary?.cumulative_energy_savings?.toFixed(1)/1000).toFixed(2) || "—"}
          unit="MW"
          color="green"
        />
        <Stat
          icon={TrendingUp}
          label="Cumulative Profit"
          value={kpiStats?.summary?.cumulative_profit?.toFixed(0) || "—"}
          unit="$/h"
          color="amber"
        />
      </div>

      {/* Product flows — vertical bar chart */}
      <div className="glass-card p-5">
        <h3 className="text-lg font-semibold text-white mb-3">Product Flows</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart
            data={Object.entries(flows)
              .filter(([name]) => name !== "Vac_residue")
              .map(([name, flow]) => ({
                name: PRODUCT_LABELS[name] || name,
                flow: parseFloat(flow.toFixed(1)),
                fill: PRODUCT_COLORS[name] || "#6b7280",
              }))}
            margin={{ top: 4, right: 8, bottom: 42, left: 48 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="name"
              stroke="#475569"
              tick={{ fontSize: 9, fill: "#94a3b8" }}
              angle={-40}
              textAnchor="end"
              interval={0}
            />
            <YAxis
              stroke="#475569"
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              tickFormatter={(v) => `${v}`}
              label={{ value: "kg/h", angle: -90, position: "insideLeft", offset: -34, fill: "#475569", fontSize: 10 }}
            />
            <Tooltip
              contentStyle={ttStyle}
              formatter={(v) => [`${v} kg/h`, "Flow"]}
            />
            <Bar dataKey="flow" radius={[3, 3, 0, 0]}>
              {Object.keys(flows)
                .filter((name) => name !== "Vac_residue")
                .map((name, i) => (
                  <Cell key={i} fill={PRODUCT_COLORS[name] || "#6b7280"} fillOpacity={0.85} />
                ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Optimization Scope */}
      {scope && (
        <div
          className="glass-card p-6"
          style={{
            border: `1px solid ${scope.no_improvement ? "rgba(100,116,139,0.25)" : "rgba(22,242,194,0.18)"}`,
          }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <Brain size={20} className="text-blue-400" />
              Optimization Scope
            </h3>
            <span className="text-xs text-gray-500">
              {scope.model_info?.algorithm || "SAC"} Agent Recommendation
            </span>
          </div>

          {/* No-improvement notice */}
          {scope.no_improvement && (
            <p className="text-sm text-gray-500 italic mb-4">
              Current operating point is at or near the agent's optimum — no improvement found this cycle.
              Retrying on next poll.
            </p>
          )}

          {/* Profit row — always shown; right column freezes at best-ever profit */}
          <div className="grid grid-cols-3 gap-4 mb-5">
            <div className="text-center">
              <p className="text-xs text-gray-500 mb-1">Current Est. Profit</p>
              <p className="text-xl font-mono font-bold text-gray-300">
                ${scope.baseline_profit?.toFixed(0)}/h
              </p>
            </div>
            <div className="flex flex-col items-center justify-center">
              {scope.no_improvement ? (
                <>
                  <p className="text-2xl font-bold text-gray-600">—</p>
                  <p className="text-xs text-gray-600">no new delta</p>
                </>
              ) : (
                <>
                  <p className="text-2xl font-bold" style={{ color: "rgb(22,242,194)" }}>
                    +{scope.delta_profit?.toFixed(0)}
                  </p>
                  <p className="text-xs" style={{ color: "rgb(22,242,194)" }}>
                    +{scope.delta_pct?.toFixed(1)}%
                  </p>
                </>
              )}
            </div>
            <div className="text-center">
              <p className="text-xs text-gray-500 mb-1">Best Est. Profit Achieved</p>
              <p
                className="text-xl font-mono font-bold"
                style={{ color: bestOptimizedProfit != null ? "rgb(22,242,194)" : undefined }}
              >
                {bestOptimizedProfit != null
                  ? `$${bestOptimizedProfit.toFixed(0)}/h`
                  : `$${scope.baseline_profit?.toFixed(0)}/h`}
              </p>
            </div>
          </div>

          {/* Per-product uplift — only when a genuine improvement exists */}
          {!scope.no_improvement && scope.product_delta && (
            <div className="mb-4">
              <p className="text-xs text-gray-500 mb-2">Per-Product Revenue Uplift ($/h)</p>
              <div className="space-y-1.5">
                {Object.entries(scope.product_delta)
                  .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                  .slice(0, 6)
                  .map(([prod, delta]) => {
                    const maxAbs = Math.max(...Object.values(scope.product_delta).map(Math.abs), 1);
                    const barColor = delta >= 0 ? "rgb(22,242,194)" : "rgb(56,152,196)";
                    return (
                      <div key={prod} className="flex items-center gap-2 text-xs">
                        <span className="w-28 text-gray-400 truncate">{prod.replace(/_/g, " ")}</span>
                        <div className="flex-1 h-2.5 bg-gray-800 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.min(Math.abs(delta) / maxAbs * 100, 100)}%`,
                              backgroundColor: barColor,
                            }}
                          />
                        </div>
                        <span className="w-16 text-right font-mono" style={{ color: barColor }}>
                          {delta >= 0 ? "+" : ""}{delta.toFixed(1)}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Footer */}
          <div className="text-xs text-gray-600 flex justify-between border-t border-gray-800 pt-2">
            <span>Model: {scope.model_info?.algorithm || "SAC"}</span>
            {scope.model_info?.best_reward != null && (
              <span>Best reward: {scope.model_info.best_reward.toFixed(3)}</span>
            )}
          </div>
        </div>
      )}

      {/* Operational Targets */}
      {(scope || lastRecommendations) && (
        <div
          className="glass-card p-6"
          style={{ border: "1px solid rgba(56,152,196,0.2)" }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <Target size={20} style={{ color: "rgb(56,152,196)" }} />
              Operational Targets
            </h3>
            {scope?.no_improvement && lastRecommendations && (
              <span className="text-xs text-gray-500 bg-gray-800/60 px-2 py-1 rounded-full">
                At optimum — holding last targets
              </span>
            )}
          </div>

          {lastRecommendations ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
              {Object.entries(lastRecommendations).map(([key, value]) => {
                const meta = ACTION_LABELS[key] || { label: key.replace(/_/g, " "), unit: "" };
                return (
                  <div
                    key={key}
                    className="bg-gray-800/50 rounded-lg px-3 py-2.5 flex flex-col gap-0.5"
                  >
                    <span className="text-xs text-gray-500">{meta.label}</span>
                    <div className="flex items-baseline gap-1">
                      <span
                        className="text-base font-mono font-semibold"
                        style={{ color: "rgb(56,152,196)" }}
                      >
                        {meta.unit === "—" ? value.toFixed(4) : value.toFixed(1)}
                      </span>
                      <span className="text-xs text-gray-600">{meta.unit}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-sm text-gray-600 italic">
              No recommendation yet — run the agent to generate targets.
            </p>
          )}
        </div>
      )}

      {/* Training Analytics */}
      {(rewardHistory.length > 0 || qHistory.length > 0 || gradHistory.length > 0 || actionDist) && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Brain size={20} className="text-blue-400" />
            Training Analytics
            {progress?.status === "training" && (
              <span className="text-xs font-normal text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded-full">
                Live · Ep {progress.episode} · Step {progress.current_step?.toLocaleString()}
              </span>
            )}
          </h3>

          {/* Rewards curve — shown after episode 20 */}
          {rewardHistory.length > 0 && rewardHistory[rewardHistory.length - 1]?.episode >= 20 && (
            <ChartCard title="Reward Curve" icon={TrendingUp}>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={rewardHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="step" stroke="#475569" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#475569" tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={ttStyle} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="avg" stroke={C.green} dot={false} strokeWidth={2} name="Avg (100)" />
                  <Line type="monotone" dataKey="best" stroke={C.purple} dot={false} strokeWidth={1} strokeDasharray="5 5" name="Best" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          )}

          {/* Q-Value / Entropy / Alpha + Gradient Norms side by side */}
          {(qHistory.length > 1 || gradHistory.length > 1) && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {qHistory.length > 1 && (
                <ChartCard title="Q-Value · Entropy · Alpha (α)" icon={Gauge}>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={qHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="step" stroke="#475569" tick={{ fontSize: 10 }} />
                      <YAxis yAxisId="left" stroke="#475569" tick={{ fontSize: 10 }} />
                      <YAxis yAxisId="right" orientation="right" stroke="#475569" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={ttStyle} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line yAxisId="left" type="monotone" dataKey="q" stroke={C.cyan} dot={false} strokeWidth={1.5} name="Mean Q" />
                      <Line yAxisId="left" type="monotone" dataKey="entropy" stroke={C.yellow} dot={false} strokeWidth={1.5} name="Entropy" />
                      <Line yAxisId="right" type="monotone" dataKey="alpha" stroke={C.pink} dot={false} strokeWidth={1} strokeDasharray="4 4" name="Alpha (α)" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>
              )}

              {gradHistory.length > 1 && (
                <ChartCard title="Gradient Norms" icon={Activity}>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={gradHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="step" stroke="#475569" tick={{ fontSize: 10 }} />
                      <YAxis stroke="#475569" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={ttStyle} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="actor" stroke={C.blue} dot={false} strokeWidth={1.5} name="Actor ∇" />
                      <Line type="monotone" dataKey="critic" stroke={C.orange} dot={false} strokeWidth={1.5} name="Critic ∇" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>
              )}
            </div>
          )}

          {/* Action Distribution */}
          {actionDist && (
            <ChartCard title="Action Distribution (latest)" icon={BarChart3}>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart
                  data={actionDist.names.map((name, i) => ({
                    name: name.replace("_draw_temp", "").replace(/_/g, " "),
                    mean: actionDist.means[i],
                    std: actionDist.stds[i],
                  }))}
                  barGap={4}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#475569" tick={{ fontSize: 9 }} />
                  <YAxis stroke="#475569" tick={{ fontSize: 10 }} domain={[-1.3, 1.3]} />
                  <Tooltip contentStyle={ttStyle} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="mean" name="Mean" radius={[4, 4, 0, 0]}>
                    {actionDist.names.map((_, i) => (
                      <Cell key={i} fill={ACTION_COLORS[i % ACTION_COLORS.length]} fillOpacity={0.85} />
                    ))}
                  </Bar>
                  <Bar dataKey="std" name="Std Dev" fill="#475569" radius={[4, 4, 0, 0]} fillOpacity={0.5} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          )}
        </div>
      )}

    </div>
  );
}
