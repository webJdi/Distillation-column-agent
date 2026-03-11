import { useState, useEffect, useCallback, useRef } from "react";
import {
  Brain,
  Play,
  Square,
  Download,
  Settings,
  TrendingUp,
  Activity,
  BarChart3,
  Gauge,
  Database,
  Zap,
  RefreshCw,
  Save,
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
  ReferenceLine,
} from "recharts";
import toast from "react-hot-toast";
import {
  startTraining,
  stopTraining,
  trainingStatus,
  listCheckpoints,
  loadCheckpoint,
  getLatestMetrics,
} from "../api";
import { useTrainingWebSocket } from "../useWebSocket";
import {
  saveTrainingRun,
  saveTrainingMetrics,
} from "../firebase";

/* ── colour palette ──────────────────────────────────────────────── */
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

/* ── small reusable card ─────────────────────────────────────────── */
function MetricCard({ label, value, sub, color = "text-white", icon: Icon }) {
  return (
    <div className="bg-gray-800/60 rounded-xl px-4 py-3 border border-gray-700/50">
      <div className="flex items-center gap-1.5 mb-1">
        {Icon && <Icon size={13} className="text-gray-500" />}
        <p className="text-[11px] text-gray-500 uppercase tracking-wider">{label}</p>
      </div>
      <p className={`text-lg font-mono font-semibold ${color}`}>
        {value ?? "—"}
      </p>
      {sub && <p className="text-[10px] text-gray-600 mt-0.5">{sub}</p>}
    </div>
  );
}

/* ── small chart wrapper ─────────────────────────────────────────── */
function ChartCard({ title, icon: Icon, children, className = "" }) {
  return (
    <div className={`glass-card p-4 ${className}`}>
      <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
        {Icon && <Icon size={15} className="text-gray-500" />}
        {title}
      </h4>
      {children}
    </div>
  );
}

/* ── tooltip style ───────────────────────────────────────────────── */
const ttStyle = {
  backgroundColor: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 8,
  color: "#e2e8f0",
  fontSize: 12,
};

/* ── main page ───────────────────────────────────────────────────── */
export default function TrainingPage() {
  const { progress, connected } = useTrainingWebSocket();
  const [config, setConfig] = useState({
    algorithm: "SAC",
    total_timesteps: 50000,
    learning_rate: 0.0003,
    batch_size: 256,
    gamma: 0.99,
    use_curriculum: true,
    scenario_name: "default",
  });
  const [status, setStatus] = useState("idle");
  const [checkpoints, setCheckpoints] = useState([]);
  const [showConfig, setShowConfig] = useState(false);

  /* detailed history arrays ─────────────────────────────────────── */
  const [rewardHistory, setRewardHistory] = useState([]);
  const [lossHistory, setLossHistory] = useState([]);
  const [qHistory, setQHistory] = useState([]);
  const [gradHistory, setGradHistory] = useState([]);
  const [bufferHistory, setBufferHistory] = useState([]);
  const [actionDist, setActionDist] = useState(null);

  /* latest scalar values (for metric cards) ─────────────────────── */
  const latest = useRef({});

  /* ── initial data load ─────────────────────────────────────────── */
  useEffect(() => {
    trainingStatus()
      .then((r) => setStatus(r.data?.status || "idle"))
      .catch(() => {});
    listCheckpoints()
      .then((r) => setCheckpoints(r.data || []))
      .catch(() => {});
    // Try to load metrics from a previous run
    getLatestMetrics()
      .then((r) => {
        const hist = r.data?.metrics_history || [];
        if (hist.length > 0) _ingestFullHistory(hist);
      })
      .catch(() => {});
  }, []);

  /* ── ingest a full metrics history array (after load/completion) ── */
  const _ingestFullHistory = useCallback((history) => {
    const rh = [], lh = [], qh = [], gh = [], bh = [];
    history.forEach((m) => {
      const step = m.current_step || 0;
      if (m.episode > 0) {
        rh.push({
          step,
          episode: m.episode,
          reward: m.episode_reward,
          avg: m.avg_reward,
          best: m.best_reward,
        });
      }
      if (m.critic_loss || m.actor_loss) {
        lh.push({ step, critic: m.critic_loss, actor: m.actor_loss });
      }
      if (m.mean_q_value !== undefined || m.entropy !== undefined) {
        qh.push({
          step,
          q: m.mean_q_value,
          entropy: m.entropy,
          alpha: m.ent_coef,
        });
      }
      if (m.actor_grad_norm !== undefined || m.critic_grad_norm !== undefined) {
        gh.push({
          step,
          actor: m.actor_grad_norm,
          critic: m.critic_grad_norm,
        });
      }
      if (m.replay_buffer_size !== undefined) {
        bh.push({
          step,
          size: m.replay_buffer_size,
          capacity: m.replay_buffer_capacity,
          pct: m.replay_buffer_pct,
        });
      }
    });
    setRewardHistory(rh);
    setLossHistory(lh);
    setQHistory(qh);
    setGradHistory(gh);
    setBufferHistory(bh);
    const last = history[history.length - 1];
    if (last) {
      latest.current = last;
      if (last.action_distribution) setActionDist(last.action_distribution);
    }
  }, []);

  /* ── ingest live progress from WebSocket ───────────────────────── */
  useEffect(() => {
    if (!progress) return;
    const m = progress;
    latest.current = m;
    setStatus(m.status);

    const step = m.current_step || 0;

    if (m.episode > 0) {
      setRewardHistory((h) => {
        const last = h[h.length - 1];
        if (last?.step === step) return h;
        const next = [
          ...h,
          {
            step,
            episode: m.episode,
            reward: m.episode_reward,
            avg: m.avg_reward,
            best: m.best_reward,
          },
        ];
        return next.length > 600 ? next.slice(-600) : next;
      });
    }

    if (m.critic_loss || m.actor_loss) {
      setLossHistory((h) => {
        const last = h[h.length - 1];
        if (last?.step === step) return h;
        const next = [
          ...h,
          { step, critic: m.critic_loss, actor: m.actor_loss },
        ];
        return next.length > 600 ? next.slice(-600) : next;
      });
    }

    if (m.mean_q_value !== undefined || m.entropy !== undefined) {
      setQHistory((h) => {
        const last = h[h.length - 1];
        if (last?.step === step) return h;
        const next = [
          ...h,
          { step, q: m.mean_q_value, entropy: m.entropy, alpha: m.ent_coef },
        ];
        return next.length > 600 ? next.slice(-600) : next;
      });
    }

    if (m.actor_grad_norm !== undefined || m.critic_grad_norm !== undefined) {
      setGradHistory((h) => {
        const last = h[h.length - 1];
        if (last?.step === step) return h;
        const next = [
          ...h,
          { step, actor: m.actor_grad_norm, critic: m.critic_grad_norm },
        ];
        return next.length > 600 ? next.slice(-600) : next;
      });
    }

    if (m.replay_buffer_size !== undefined) {
      setBufferHistory((h) => {
        const last = h[h.length - 1];
        if (last?.step === step) return h;
        const next = [
          ...h,
          {
            step,
            size: m.replay_buffer_size,
            capacity: m.replay_buffer_capacity,
            pct: m.replay_buffer_pct,
          },
        ];
        return next.length > 600 ? next.slice(-600) : next;
      });
    }

    if (m.action_distribution) setActionDist(m.action_distribution);

    // On training completion, save to Firebase
    if (m.status === "completed" && m.run_id) {
      _saveToFirebase(m);
    }
  }, [progress]);

  /* ── save to Firebase on completion ───────────────────────────── */
  const _saveToFirebase = useCallback(async (finalMetrics) => {
    try {
      await saveTrainingRun({
        run_id: finalMetrics.run_id,
        algorithm: finalMetrics.config?.algorithm || config.algorithm,
        total_timesteps: finalMetrics.total_steps,
        best_reward: finalMetrics.best_reward,
        avg_reward: finalMetrics.avg_reward,
        episodes: finalMetrics.episode,
        training_time: finalMetrics.training_time_seconds,
        critic_loss: finalMetrics.critic_loss,
        actor_loss: finalMetrics.actor_loss,
        mean_q_value: finalMetrics.mean_q_value,
        entropy: finalMetrics.entropy,
        ent_coef: finalMetrics.ent_coef,
        checkpoint_path: finalMetrics.checkpoint_path,
        checkpoint_size_mb: finalMetrics.checkpoint_size_mb,
        config: finalMetrics.config,
      });
      // Also save full metrics history
      const metricsResp = await getLatestMetrics();
      const hist = metricsResp.data?.metrics_history || [];
      if (hist.length > 0) {
        await saveTrainingMetrics(finalMetrics.run_id, finalMetrics, hist);
      }
      toast.success("Training metrics saved to Firebase");
    } catch (err) {
      console.warn("Firebase save failed:", err);
    }
  }, [config]);

  /* ── handlers ──────────────────────────────────────────────────── */
  const handleStart = async () => {
    try {
      await startTraining(config);
      setStatus("training");
      setRewardHistory([]);
      setLossHistory([]);
      setQHistory([]);
      setGradHistory([]);
      setBufferHistory([]);
      setActionDist(null);
      latest.current = {};
      toast.success("Training started");
    } catch (err) {
      toast.error(err.response?.data?.detail || "Failed to start");
    }
  };

  const handleStop = async () => {
    try {
      await stopTraining();
      setStatus("idle");
      toast.success("Training stopped");
    } catch {
      toast.error("Failed to stop");
    }
  };

  const handleLoadCheckpoint = async (cp) => {
    try {
      await loadCheckpoint(cp.path);
      toast.success("Checkpoint loaded");
      // Load associated metrics
      const name = cp.name;
      const resp = await getLatestMetrics();
      const hist = resp.data?.metrics_history || [];
      if (hist.length > 0) _ingestFullHistory(hist);
    } catch {
      toast.error("Failed to load checkpoint");
    }
  };

  const handleRefreshCheckpoints = async () => {
    const r = await listCheckpoints();
    setCheckpoints(r.data || []);
  };

  const isTraining = status === "training";
  const m = latest.current;
  const pct =
    m?.total_steps > 0
      ? ((m.current_step / m.total_steps) * 100).toFixed(1)
      : 0;

  const hasChartData =
    rewardHistory.length > 0 ||
    lossHistory.length > 0 ||
    qHistory.length > 0;

  /* ── render ────────────────────────────────────────────────────── */
  return (
    <div className="space-y-5 max-w-7xl">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">RL Agent Training</h2>
          <p className="text-sm text-gray-500 mt-1">
            Train &amp; monitor SAC / PPO / TD3 agent on the CDU + VDU simulation
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span
            className={`h-2 w-2 rounded-full ${
              connected ? "bg-green-400" : "bg-red-500"
            }`}
            title={connected ? "WebSocket connected" : "WebSocket disconnected"}
          />
          <button
            onClick={() => setShowConfig(!showConfig)}
            className="p-2 rounded-lg bg-gray-800 text-gray-400 hover:text-white transition"
          >
            <Settings size={18} />
          </button>
          {isTraining ? (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white font-medium transition"
            >
              <Square size={16} /> Stop
            </button>
          ) : (
            <button
              onClick={handleStart}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium transition"
            >
              <Play size={16} /> Start Training
            </button>
          )}
        </div>
      </div>

      {/* ── Config panel ───────────────────────────────────────────── */}
      {showConfig && (
        <div className="glass-card p-5 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 animate-fade-in">
          {[
            { key: "algorithm", label: "Algorithm", type: "select", options: ["SAC", "PPO", "TD3"] },
            { key: "total_timesteps", label: "Total Steps", type: "number", min: 1000, max: 1000000, step: 10000 },
            { key: "learning_rate", label: "Learning Rate", type: "number", min: 0.000001, max: 0.1, step: 0.0001 },
            { key: "batch_size", label: "Batch Size", type: "number", min: 32, max: 2048, step: 32 },
            { key: "gamma", label: "Gamma (γ)", type: "number", min: 0.9, max: 0.999, step: 0.001 },
            { key: "scenario_name", label: "Price Scenario", type: "text" },
          ].map(({ key, label, type, options, ...rest }) => (
            <div key={key}>
              <label className="block text-xs text-gray-400 mb-1">{label}</label>
              {type === "select" ? (
                <select
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white outline-none focus:ring-2 focus:ring-blue-500"
                  value={config[key]}
                  onChange={(e) => setConfig((c) => ({ ...c, [key]: e.target.value }))}
                >
                  {options.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              ) : (
                <input
                  type={type}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white font-mono outline-none focus:ring-2 focus:ring-blue-500"
                  value={config[key]}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      [key]: type === "number" ? Number(e.target.value) : e.target.value,
                    }))
                  }
                  {...rest}
                />
              )}
            </div>
          ))}
          <div className="flex items-end">
            <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
              <input
                type="checkbox"
                checked={config.use_curriculum}
                onChange={(e) =>
                  setConfig((c) => ({ ...c, use_curriculum: e.target.checked }))
                }
                className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500"
              />
              Curriculum Learning
            </label>
          </div>
        </div>
      )}

      {/* ── Progress bar ───────────────────────────────────────────── */}
      {(isTraining || status === "completed") && m?.current_step > 0 && (
        <div
          className={`glass-card p-5 animate-fade-in ${
            isTraining ? "ring-1 ring-blue-500/20" : "ring-1 ring-green-500/20"
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span
              className={`text-sm font-medium flex items-center gap-2 ${
                isTraining ? "text-blue-400" : "text-green-400"
              }`}
            >
              <Brain size={16} className={isTraining ? "animate-pulse" : ""} />
              {isTraining ? "Training…" : "Completed"}
              {m.training_time_seconds && (
                <span className="text-gray-500 text-xs ml-2">
                  {m.training_time_seconds.toFixed(1)}s
                </span>
              )}
            </span>
            <span className="text-sm font-mono text-gray-400">{pct}%</span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-300 ${
                isTraining ? "bg-blue-500" : "bg-green-500"
              }`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}

      {/* ── Metric cards grid ──────────────────────────────────────── */}
      {(hasChartData || m?.current_step > 0) && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <MetricCard
            label="Step"
            value={m?.current_step?.toLocaleString() || 0}
            sub={`/ ${(m?.total_steps || 0).toLocaleString()}`}
            icon={Activity}
          />
          <MetricCard
            label="Episodes"
            value={m?.episode || 0}
            icon={RefreshCw}
          />
          <MetricCard
            label="Avg Reward"
            value={m?.avg_reward?.toFixed(4) || "—"}
            color="text-green-400"
            icon={TrendingUp}
          />
          <MetricCard
            label="Best Reward"
            value={m?.best_reward?.toFixed(4) || "—"}
            color="text-purple-400"
            icon={Zap}
          />
          <MetricCard
            label="Critic Loss"
            value={m?.critic_loss?.toFixed(4) || "—"}
            color="text-orange-400"
            icon={Activity}
          />
          <MetricCard
            label="Actor Loss"
            value={m?.actor_loss?.toFixed(4) || "—"}
            color="text-red-400"
            icon={Activity}
          />
          <MetricCard
            label="Mean Q-Value"
            value={m?.mean_q_value?.toFixed(4) || "—"}
            color="text-cyan-400"
            icon={Gauge}
          />
          <MetricCard
            label="Entropy"
            value={m?.entropy?.toFixed(4) || "—"}
            color="text-yellow-400"
            icon={BarChart3}
          />
          <MetricCard
            label="Alpha (α)"
            value={m?.ent_coef?.toFixed(5) || "—"}
            color="text-pink-400"
            icon={Settings}
          />
          <MetricCard
            label="Actor ∇"
            value={m?.actor_grad_norm?.toFixed(4) || "—"}
            sub="grad norm"
            color="text-blue-400"
            icon={TrendingUp}
          />
          <MetricCard
            label="Critic ∇"
            value={m?.critic_grad_norm?.toFixed(4) || "—"}
            sub="grad norm"
            color="text-orange-400"
            icon={TrendingUp}
          />
          <MetricCard
            label="Buffer"
            value={
              m?.replay_buffer_size != null
                ? `${(m.replay_buffer_size / 1000).toFixed(1)}k`
                : "—"
            }
            sub={
              m?.replay_buffer_pct != null
                ? `${m.replay_buffer_pct}% full`
                : undefined
            }
            color="text-slate-300"
            icon={Database}
          />
        </div>
      )}

      {/* ── Reward curve (full width) ──────────────────────────────── */}
      {rewardHistory.length > 1 && (
        <ChartCard title="Reward Curve" icon={TrendingUp}>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={rewardHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="step" stroke="#475569" tick={{ fontSize: 10 }} />
              <YAxis stroke="#475569" tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={ttStyle} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="reward" stroke={C.blue} dot={false} strokeWidth={1} name="Episode" />
              <Line type="monotone" dataKey="avg" stroke={C.green} dot={false} strokeWidth={2} name="Avg (100)" />
              <Line type="monotone" dataKey="best" stroke={C.purple} dot={false} strokeWidth={1} strokeDasharray="5 5" name="Best" />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      )}

      {/* ── Loss + Q-Value row (2 columns) ─────────────────────────── */}
      {(lossHistory.length > 1 || qHistory.length > 1) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* Critic & Actor Loss */}
          {lossHistory.length > 1 && (
            <ChartCard title="Critic & Actor Loss" icon={Activity}>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={lossHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="step" stroke="#475569" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#475569" tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={ttStyle} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="critic" stroke={C.orange} dot={false} strokeWidth={1.5} name="Critic Loss" />
                  <Line type="monotone" dataKey="actor" stroke={C.red} dot={false} strokeWidth={1.5} name="Actor Loss" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          )}

          {/* Mean Q-Value, Entropy & Alpha */}
          {qHistory.length > 1 && (
            <ChartCard title="Q-Value, Entropy & Alpha" icon={Gauge}>
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
        </div>
      )}

      {/* ── Gradient Norms + Replay Buffer row ─────────────────────── */}
      {(gradHistory.length > 1 || bufferHistory.length > 1) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* Gradient Norms */}
          {gradHistory.length > 1 && (
            <ChartCard title="Gradient Norms" icon={TrendingUp}>
              <ResponsiveContainer width="100%" height={200}>
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

          {/* Replay Buffer utilisation */}
          {bufferHistory.length > 1 && (
            <ChartCard title="Replay Buffer Size" icon={Database}>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={bufferHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="step" stroke="#475569" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#475569" tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={ttStyle}
                    formatter={(v) => [(v / 1000).toFixed(1) + "k", ""]}
                  />
                  <Line type="monotone" dataKey="size" stroke={C.slate} dot={false} strokeWidth={2} name="Buffer Size" />
                  {bufferHistory[0]?.capacity && (
                    <ReferenceLine
                      y={bufferHistory[0].capacity}
                      stroke={C.red}
                      strokeDasharray="4 4"
                      label={{ value: "Capacity", fill: "#ef4444", fontSize: 10 }}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          )}
        </div>
      )}

      {/* ── Action Distribution ─────────────────────────────────────── */}
      {actionDist && (
        <ChartCard title="Action Distribution (latest)" icon={BarChart3}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart
              data={actionDist.names.map((name, i) => ({
                name: name.replace("_draw_temp", "").replace("_", " "),
                mean: actionDist.means[i],
                std: actionDist.stds[i],
              }))}
              barGap={4}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" stroke="#475569" tick={{ fontSize: 10 }} />
              <YAxis stroke="#475569" tick={{ fontSize: 10 }} />
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

      {/* ── Live Training Progress Box ─────────────────────────────── */}
      {isTraining && (
        <div className="glass-card p-6 ring-1 ring-blue-500/30 animate-fade-in">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <Brain size={20} className="text-blue-400 animate-pulse" />
              Training in Progress
            </h3>
            <span className="px-3 py-1 rounded-full bg-blue-600/20 text-blue-400 text-xs font-semibold uppercase tracking-wider">
              {config.algorithm}
            </span>
          </div>

          {/* Progress bar */}
          <div className="mb-4">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-sm text-gray-400">
                Episode {m?.episode || 0} &middot; Step{" "}
                {(m?.current_step || 0).toLocaleString()} /{" "}
                {(m?.total_steps || config.total_timesteps).toLocaleString()}
              </span>
              <span className="text-sm font-mono text-blue-400 font-semibold">{pct}%</span>
            </div>
            <div className="h-2.5 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-500"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>

          {/* Key metrics row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
            <div className="bg-gray-800/60 rounded-lg px-3 py-2">
              <p className="text-[10px] text-gray-500 uppercase">Avg Reward</p>
              <p className="text-lg font-mono font-bold text-green-400">
                {m?.avg_reward?.toFixed(4) ?? "—"}
              </p>
            </div>
            <div className="bg-gray-800/60 rounded-lg px-3 py-2">
              <p className="text-[10px] text-gray-500 uppercase">Best Reward</p>
              <p className="text-lg font-mono font-bold text-purple-400">
                {m?.best_reward?.toFixed(4) ?? "—"}
              </p>
            </div>
            <div className="bg-gray-800/60 rounded-lg px-3 py-2">
              <p className="text-[10px] text-gray-500 uppercase">Critic Loss</p>
              <p className="text-lg font-mono font-bold text-orange-400">
                {m?.critic_loss?.toFixed(4) ?? "—"}
              </p>
            </div>
            <div className="bg-gray-800/60 rounded-lg px-3 py-2">
              <p className="text-[10px] text-gray-500 uppercase">Entropy</p>
              <p className="text-lg font-mono font-bold text-yellow-400">
                {m?.entropy?.toFixed(4) ?? "—"}
              </p>
            </div>
          </div>

          {/* Inline reward curve */}
          {rewardHistory.length > 1 && (
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Live Reward Curve</p>
              <ResponsiveContainer width="100%" height={140}>
                <LineChart data={rewardHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="step" stroke="#475569" tick={{ fontSize: 9 }} />
                  <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
                  <Tooltip contentStyle={ttStyle} />
                  <Line type="monotone" dataKey="avg" stroke={C.green} dot={false} strokeWidth={2} name="Avg" />
                  <Line type="monotone" dataKey="reward" stroke={C.blue} dot={false} strokeWidth={1} opacity={0.5} name="Episode" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* ── Checkpoints ─────────────────────────────────────────────── */}
      <div className="glass-card p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Download size={18} className="text-gray-400" />
            Saved Checkpoints
          </h3>
          <button
            onClick={handleRefreshCheckpoints}
            className="p-1.5 rounded-lg bg-gray-800 text-gray-400 hover:text-white transition"
            title="Refresh"
          >
            <RefreshCw size={14} />
          </button>
        </div>
        {checkpoints.length === 0 ? (
          <p className="text-sm text-gray-500">
            No checkpoints yet. Train the agent to create one.
          </p>
        ) : (
          <div className="space-y-2">
            {checkpoints.map((cp) => (
              <div
                key={cp.name}
                className="flex items-center justify-between bg-gray-800/50 rounded-lg px-4 py-3"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-mono text-white truncate">
                    {cp.name}
                  </p>
                  <p className="text-xs text-gray-500">
                    {cp.size_mb?.toFixed(1)} MB &middot;{" "}
                    {new Date(cp.created).toLocaleString()}
                  </p>
                  {cp.metrics_summary && (
                    <div className="flex gap-3 mt-1 text-[10px] text-gray-500">
                      <span>
                        Best: <b className="text-purple-400">{cp.metrics_summary.best_reward?.toFixed(3)}</b>
                      </span>
                      <span>
                        Avg: <b className="text-green-400">{cp.metrics_summary.avg_reward?.toFixed(3)}</b>
                      </span>
                      <span>
                        Episodes: <b className="text-gray-300">{cp.metrics_summary.episodes}</b>
                      </span>
                      {cp.metrics_summary.critic_loss != null && (
                        <span>
                          Critic: <b className="text-orange-400">{cp.metrics_summary.critic_loss?.toFixed(3)}</b>
                        </span>
                      )}
                      {cp.metrics_summary.mean_q_value != null && (
                        <span>
                          Q: <b className="text-cyan-400">{cp.metrics_summary.mean_q_value?.toFixed(3)}</b>
                        </span>
                      )}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => handleLoadCheckpoint(cp)}
                  className="ml-3 px-3 py-1.5 rounded-lg bg-blue-600/20 text-blue-400 text-xs font-medium hover:bg-blue-600/30 transition"
                >
                  Load
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
