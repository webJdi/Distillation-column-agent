/**
 * Firebase configuration for CDU Optimizer.
 * Firestore stores prices, training metrics, model metadata, checkpoints, and run history.
 */
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import {
  getFirestore,
  doc,
  setDoc,
  getDoc,
  getDocs,
  deleteDoc,
  collection,
  query,
  orderBy,
  limit,
  serverTimestamp,
} from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyCuapOt6gWnhUztduXZLGAO9Cp0Mf7PcPI",
  authDomain: "cdu-optimizer.firebaseapp.com",
  projectId: "cdu-optimizer",
  storageBucket: "cdu-optimizer.firebasestorage.app",
  messagingSenderId: "235502423586",
  appId: "1:235502423586:web:c36d60b5e8bd44705c447b",
  measurementId: "G-M4QHVCNX04",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
let analytics = null;
try {
  analytics = getAnalytics(app);
} catch {
  /* analytics may fail in dev/localhost */
}
const db = getFirestore(app);

// ── Firestore helpers — Training ───────────────────────────────────────────

/**
 * Save a training run's metrics to Firestore.
 */
export async function saveTrainingRun(runData) {
  const runId = runData.run_id || `run_${Date.now()}`;
  const docRef = doc(db, "training_runs", runId);
  await setDoc(docRef, {
    ...runData,
    savedAt: serverTimestamp(),
  });
  return runId;
}

/**
 * Save detailed training metrics history to Firestore.
 * Stores the final metrics + last 200 history points (Firestore 1MB limit).
 */
export async function saveTrainingMetrics(runId, finalMetrics, metricsHistory) {
  const docRef = doc(db, "training_metrics", `metrics_${runId}`);
  await setDoc(docRef, {
    run_id: runId,
    final_metrics: finalMetrics,
    metrics_history: (metricsHistory || []).slice(-200),
    metrics_count: (metricsHistory || []).length,
    savedAt: serverTimestamp(),
  });
  return runId;
}

/**
 * Get a training run from Firestore.
 */
export async function getTrainingRun(runId) {
  const docRef = doc(db, "training_runs", runId);
  const snap = await getDoc(docRef);
  return snap.exists() ? snap.data() : null;
}

/**
 * Get training metrics from Firestore.
 */
export async function getTrainingMetrics(runId) {
  const docRef = doc(db, "training_metrics", `metrics_${runId}`);
  const snap = await getDoc(docRef);
  return snap.exists() ? snap.data() : null;
}

/**
 * List recent training runs from Firestore.
 */
export async function listTrainingRuns(maxResults = 20) {
  try {
    const q = query(
      collection(db, "training_runs"),
      orderBy("savedAt", "desc"),
      limit(maxResults)
    );
    const snap = await getDocs(q);
    return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
  } catch {
    return [];
  }
}

// ── Firestore helpers — Prices ─────────────────────────────────────────────

/**
 * Save prices for a scenario to Firestore.
 */
export async function savePricesToFirestore(scenarioName, prices) {
  const docId = `prices_${scenarioName}`;
  const docRef = doc(db, "prices", docId);
  await setDoc(docRef, {
    scenario_name: scenarioName,
    prices,
    updated_at: new Date().toISOString(),
    savedAt: serverTimestamp(),
  });
  return docId;
}

/**
 * Get prices for a scenario from Firestore.
 */
export async function getPricesFromFirestore(scenarioName = "default") {
  const docId = `prices_${scenarioName}`;
  const docRef = doc(db, "prices", docId);
  const snap = await getDoc(docRef);
  return snap.exists() ? snap.data() : null;
}

/**
 * List all price scenarios from Firestore.
 */
export async function listPriceScenarios() {
  try {
    const snap = await getDocs(collection(db, "prices"));
    return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
  } catch {
    return [];
  }
}

// ── Firestore helpers — Checkpoints ────────────────────────────────────────

/**
 * List all checkpoints from Firestore.
 */
export async function listCheckpointsFromFirestore() {
  try {
    const q = query(
      collection(db, "checkpoints"),
      orderBy("created_at", "desc")
    );
    const snap = await getDocs(q);
    return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
  } catch {
    return [];
  }
}

/**
 * Get checkpoint metadata from Firestore.
 */
export async function getCheckpointFromFirestore(runId) {
  const docRef = doc(db, "checkpoints", runId);
  const snap = await getDoc(docRef);
  return snap.exists() ? snap.data() : null;
}

// ── Firestore helpers — Optimization Results ───────────────────────────────

/**
 * List recent optimization results from Firestore.
 */
export async function listOptimizationResults(maxResults = 20) {
  try {
    const q = query(
      collection(db, "optimization_results"),
      orderBy("created_at", "desc"),
      limit(maxResults)
    );
    const snap = await getDocs(q);
    return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
  } catch {
    return [];
  }
}

export { app, db, analytics };
