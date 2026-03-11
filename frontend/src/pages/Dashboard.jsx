import { useEffect, useState } from "react";
import {
  Gauge,
  TrendingUp,
  Droplets,
  Zap,
  AlertTriangle,
  Brain,
} from "lucide-react";
import { trainingStatus, getProductFlows, getSimState, getD95 } from "../api";
import { useTrainingWebSocket } from "../useWebSocket";

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
        <div className={`p-2 rounded-lg bg-gray-800 ${text}`}>
          <Icon size={20} />
        </div>
        <span className="text-sm text-gray-400">{label}</span>
      </div>
      <p className="text-2xl font-bold text-white">
        {value}
        {unit && <span className="text-sm text-gray-500 ml-1">{unit}</span>}
      </p>
    </div>
  );
}

/* ── Product flow bar ─────────────────────────────────────────── */
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

function ProductBar({ name, flow, maxFlow = 1000 }) {
  const pct = Math.min((flow / maxFlow) * 100, 100);
  const color = PRODUCT_COLORS[name] || "#6b7280";
  return (
    <div className="flex items-center gap-3">
      <span className="w-16 text-xs font-mono text-gray-400 truncate">{name.replace(/_/g, " ")}</span>
      <div className="flex-1 h-5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="w-20 text-right text-sm font-mono text-gray-300">
        {flow.toFixed(1)} kg/h
      </span>
    </div>
  );
}

/* ── Dashboard page ───────────────────────────────────────────── */
export default function Dashboard() {
  const { progress, connected } = useTrainingWebSocket();
  const [status, setStatus] = useState(null);
  const [throughput, setThroughput] = useState(null);
  const [d95, setD95] = useState({});

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
  }, []);

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
            ADU + NSU + VDU real-time overview
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
        </div>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <Stat
          icon={Gauge}
          label="Agent Status"
          value={agentStatus}
          color={agentStatus === "training" ? "green" : "blue"}
        />
        <Stat icon={TrendingUp} label="Avg Reward" value={avgReward} color="purple" />
        <Stat icon={Zap} label="Episode" value={episode} color="amber" />
        <Stat icon={Droplets} label="Est. Profit" value={profit} unit="$/h" color="green" />
        <Stat
          icon={Gauge}
          label="Throughput"
          value={throughput != null ? throughput.toFixed(0) : "—"}
          unit="kg/h"
          color="red"
        />
      </div>

      {/* Product flows */}
      <div className="glass-card p-6">
        <h3 className="text-lg font-semibold text-white mb-4">
          Product Flows
        </h3>
        <div className="space-y-3">
          {(() => {
            const maxFlow = Math.max(...Object.values(flows), 1);
            return Object.entries(flows).map(([name, flow]) => (
              <ProductBar key={name} name={name} flow={flow} maxFlow={maxFlow} />
            ));
          })()}
        </div>
      </div>

      {/* D95% Estimates */}
      {Object.keys(d95).length > 0 && (
        <div className="glass-card p-6">
          <h3 className="text-lg font-semibold text-white mb-4">
            D95% Distillation Temperature
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {Object.entries(d95)
              .filter(([name]) => !["Uncondensed_Gas", "StabOffGas", "LPG", "Offgas", "SRN"].includes(name))
              .map(([name, temp]) => (
              <div
                key={name}
                className="flex items-center justify-between bg-gray-800/50 rounded-lg px-3 py-2"
              >
                <span className="text-xs text-gray-400 truncate mr-2">
                  {name.replace(/_/g, " ")}
                </span>
                <span className="text-sm font-mono text-amber-400 whitespace-nowrap">
                  {temp > 0 ? `${temp.toFixed(1)}°C` : "—"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Training progress (live) */}
      {progress && progress.status === "training" && (
        <div className="glass-card p-6 ring-1 ring-blue-500/20">
          <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
            <Brain size={20} className="text-blue-400" />
            Training in Progress
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <div>
              <p className="text-xs text-gray-500">Progress</p>
              <p className="text-sm font-mono text-white">
                {progress.current_step?.toLocaleString()} /{" "}
                {progress.total_steps?.toLocaleString()}
              </p>
              <div className="mt-1 h-1.5 bg-gray-800 rounded-full">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all"
                  style={{
                    width: `${
                      progress.total_steps
                        ? (progress.current_step / progress.total_steps) * 100
                        : 0
                    }%`,
                  }}
                />
              </div>
            </div>
            <div>
              <p className="text-xs text-gray-500">Episode</p>
              <p className="text-lg font-bold text-white">{progress.episode}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Best Reward</p>
              <p className="text-lg font-bold text-green-400">
                {progress.best_reward?.toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Avg Reward (last 100)</p>
              <p className="text-lg font-bold text-purple-400">
                {progress.avg_reward?.toFixed(2)}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Safety alerts placeholder */}
      <div className="glass-card p-6 ring-1 ring-amber-500/10">
        <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
          <AlertTriangle size={18} className="text-amber-400" />
          Safety Status
        </h3>
        <p className="text-sm text-green-400">
          All parameters within safe operating limits.
        </p>
      </div>
    </div>
  );
}
