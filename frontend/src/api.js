/** Axios instance pre-configured for the FastAPI backend. */
import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
  timeout: 30_000,
});

// ── Prices ──────────────────────────────────────────────────────
export const savePrices = (prices) => api.post("/prices/", prices);
export const getPrices = (scenario = "default") =>
  api.get(`/prices/${scenario}`);
export const listScenarios = () => api.get("/prices/scenarios");
export const createScenario = (scenario) =>
  api.post("/prices/scenario", scenario);

// ── Simulation ──────────────────────────────────────────────────
export const loadFlowsheet = () => api.post("/simulation/load");
export const getSimState = () => api.get("/simulation/state");
export const solveFlowsheet = () => api.post("/simulation/solve");
export const applyAction = (action) =>
  api.post("/simulation/apply-action", action);
export const applyDisturbance = (dist) =>
  api.post("/simulation/apply-disturbance", dist);
export const getProductFlows = () => api.get("/simulation/products");
export const getD95 = () => api.get("/simulation/d95");

// ── Training ────────────────────────────────────────────────────
export const startTraining = (config) =>
  api.post("/training/start", config);
export const stopTraining = () => api.post("/training/stop");
export const trainingStatus = () => api.get("/training/status");
export const listCheckpoints = () => api.get("/training/checkpoints");
export const loadCheckpoint = (path) =>
  api.post(`/training/load-checkpoint?path=${encodeURIComponent(path)}`);
export const optimize = (req) => api.post("/training/optimize", req);
export const trainingHistory = () => api.get("/training/history");
export const getTrainingMetrics = (runId) =>
  runId
    ? api.get(`/training/metrics/${encodeURIComponent(runId)}`)
    : api.get("/training/metrics");
export const getLatestMetrics = () => api.get("/training/metrics");

// ── Disturbance ─────────────────────────────────────────────────
export const analyzeDisturbance = (dist) =>
  api.post("/disturbance/analyze", dist);
export const disturbancePresets = () => api.get("/disturbance/presets");

// ── AI Agent ────────────────────────────────────────────────────
export const askAI = (query) => api.post("/ai/ask", query);
export const generateReport = (req) => api.post("/ai/report", req);
export const aiCapabilities = () => api.get("/ai/capabilities");
export const clearAIHistory = () => api.post("/ai/clear-history");

export default api;
