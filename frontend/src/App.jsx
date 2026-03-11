import { Routes, Route, NavLink } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import {
  Gauge,
  DollarSign,
  Brain,
  Activity,
  Flame,
  MessageSquare,
} from "lucide-react";

import Dashboard from "./pages/Dashboard";
import PricesPage from "./pages/PricesPage";
import TrainingPage from "./pages/TrainingPage";
import DisturbancePage from "./pages/DisturbancePage";
import AIAgentPage from "./pages/AIAgentPage";
import OptimizePage from "./pages/OptimizePage";

const NAV = [
  { to: "/", icon: Gauge, label: "Dashboard" },
  { to: "/prices", icon: DollarSign, label: "Prices" },
  { to: "/training", icon: Brain, label: "Training" },
  { to: "/optimize", icon: Activity, label: "Optimize" },
  { to: "/disturbance", icon: Flame, label: "Disturbance" },
  { to: "/ai", icon: MessageSquare, label: "AI Agent" },
];

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Toaster
        position="top-right"
        toastOptions={{
          style: { background: "#1f2937", color: "#f3f4f6", border: "1px solid #374151" },
        }}
      />

      {/* ── Sidebar ──────────────────────────────────────────── */}
      <aside className="w-56 shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="p-5 border-b border-gray-800">
          <h1 className="text-lg font-bold text-blue-400 tracking-tight">
            CDU Optimizer
          </h1>
          <p className="text-[11px] text-gray-500 mt-0.5">RL-Powered Distillation</p>
        </div>

        <nav className="flex-1 py-4 space-y-1 px-3">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-blue-600/20 text-blue-400"
                    : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
                }`
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-gray-800 text-[11px] text-gray-600">
          v1.0.0 &middot; DWSIM + SAC
        </div>
      </aside>

      {/* ── Main content ─────────────────────────────────────── */}
      <main className="flex-1 overflow-auto bg-gray-950 p-6">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/prices" element={<PricesPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/optimize" element={<OptimizePage />} />
          <Route path="/disturbance" element={<DisturbancePage />} />
          <Route path="/ai" element={<AIAgentPage />} />
        </Routes>
      </main>
    </div>
  );
}
