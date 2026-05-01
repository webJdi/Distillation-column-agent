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
    <div className="app-shell flex">
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "rgba(19, 29, 44, 0.9)",
            color: "#d7e9fb",
            border: "1px solid rgba(110, 138, 168, 0.35)",
            boxShadow: "0 10px 24px rgba(3, 8, 17, 0.45)",
            backdropFilter: "blur(12px)",
          },
        }}
      />

      {/* ── Sidebar ──────────────────────────────────────────── */}
      <aside className="app-sidebar w-56 shrink-0 flex flex-col">
        <div className="p-5 border-b border-gray-800/60">
          <h1 className="app-brand text-lg font-bold tracking-tight">
            Cognito
          </h1>
          <p className="app-subtitle text-[11px] mt-0.5">Deep RL Process Optimizer</p>
        </div>

        <nav className="flex-1 py-4 space-y-1 px-3">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `nav-link flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-semibold transition-colors ${
                  isActive ? "nav-link-active" : ""
                }`
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-gray-800/60 text-[11px] text-gray-600">
          v1.0.0 &middot; DWSIM + SAC
        </div>
      </aside>

      {/* ── Main content ─────────────────────────────────────── */}
      <main className="app-main flex-1 overflow-auto p-6">
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
