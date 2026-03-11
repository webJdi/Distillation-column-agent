import { useState, useRef, useEffect } from "react";
import {
  MessageSquare,
  Send,
  FileText,
  Trash2,
  Bot,
  User,
  Sparkles,
  Loader2,
} from "lucide-react";
import toast from "react-hot-toast";
import { askAI, generateReport, clearAIHistory } from "../api";

const QUICK_QUESTIONS = [
  "How does the RL agent decide the optimal reflux ratio for both ADU and VDU?",
  "Explain the CDU + VDU product streams and their cut points",
  "What happens when the feed temperature increases by 20°C?",
  "How is the reward function calculated for the 10 product streams?",
  "What safety constraints are enforced across both columns?",
  "How does curriculum learning help training?",
];

const REPORT_TYPES = [
  { value: "summary", label: "Executive Summary" },
  { value: "detailed", label: "Detailed Technical" },
  { value: "optimization", label: "Optimization Analysis" },
  { value: "comparison", label: "Scenario Comparison" },
];

function MessageBubble({ role, text }) {
  const isUser = role === "user";
  return (
    <div
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""} animate-fade-in`}
    >
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
          isUser ? "bg-blue-600" : "bg-purple-600"
        }`}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-blue-600 text-white rounded-br-md"
            : "bg-gray-800 text-gray-200 rounded-bl-md"
        }`}
      >
        {/* Render markdown-ish formatting */}
        {text.split("\n").map((line, i) => {
          if (line.startsWith("# "))
            return (
              <h3 key={i} className="text-base font-bold mt-2 mb-1">
                {line.slice(2)}
              </h3>
            );
          if (line.startsWith("## "))
            return (
              <h4 key={i} className="text-sm font-semibold mt-2 mb-1">
                {line.slice(3)}
              </h4>
            );
          if (line.startsWith("**") && line.endsWith("**"))
            return (
              <p key={i} className="font-bold">
                {line.slice(2, -2)}
              </p>
            );
          if (line.startsWith("- "))
            return (
              <li key={i} className="ml-4 list-disc">
                {line.slice(2)}
              </li>
            );
          if (line.startsWith("```"))
            return (
              <code
                key={i}
                className="block bg-gray-900 rounded px-2 py-1 font-mono text-xs my-1"
              >
                {line.slice(3)}
              </code>
            );
          if (line.startsWith("|"))
            return (
              <span key={i} className="block font-mono text-xs">
                {line}
              </span>
            );
          if (line.trim() === "") return <br key={i} />;
          return <p key={i}>{line}</p>;
        })}
      </div>
    </div>
  );
}

export default function AIAgentPage() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hello! I'm the CDU + VDU Optimizer AI Agent, powered by Google Gemini. I can explain how the RL system works across both atmospheric and vacuum columns, analyze optimization results for all 10 product streams, generate reports, and answer your questions about crude distillation. What would you like to know?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || loading) return;

    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setLoading(true);

    try {
      const r = await askAI({ question: q, include_current_state: true });
      setMessages((m) => [
        ...m,
        { role: "assistant", text: r.data?.answer || "No response." },
      ]);
    } catch {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: "Sorry, I couldn't process that right now. The backend may be offline.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleQuickQuestion = (q) => {
    setInput(q);
  };

  const handleGenerateReport = async (type) => {
    setReportLoading(true);
    setMessages((m) => [
      ...m,
      { role: "user", text: `Generate a ${type} report` },
    ]);
    try {
      const r = await generateReport({
        report_type: type,
        scenario_names: ["default"],
        include_charts: true,
      });
      const summary = r.data?.summary || r.data?.content || "Report generated.";
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: `**Report Generated: ${type.toUpperCase()}**\n\n${summary}\n\n📄 Saved to: ${r.data?.file_path || "reports/"}`,
        },
      ]);
      toast.success(`${type} report generated`);
    } catch {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: "Failed to generate report." },
      ]);
    } finally {
      setReportLoading(false);
    }
  };

  const handleClear = async () => {
    try {
      await clearAIHistory();
    } catch {}
    setMessages([
      {
        role: "assistant",
        text: "Conversation cleared. How can I help you?",
      },
    ]);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-3rem)] max-w-5xl">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Sparkles size={22} className="text-purple-400" />
            AI Agent
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Explanations, analysis, and report generation
          </p>
        </div>
        <div className="flex gap-2">
          {/* Report buttons */}
          <div className="flex gap-1">
            {REPORT_TYPES.map((rt) => (
              <button
                key={rt.value}
                onClick={() => handleGenerateReport(rt.value)}
                disabled={reportLoading}
                className="px-3 py-1.5 rounded-lg bg-gray-800 text-xs text-gray-400 hover:text-white hover:bg-gray-700 border border-gray-700 transition flex items-center gap-1"
              >
                <FileText size={12} />
                {rt.label}
              </button>
            ))}
          </div>
          <button
            onClick={handleClear}
            className="p-2 rounded-lg bg-gray-800 text-gray-400 hover:text-red-400 transition"
            title="Clear conversation"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      {/* Chat messages */}
      <div className="flex-1 overflow-auto glass-card p-5 space-y-4 mb-4">
        {messages.map((msg, i) => (
          <MessageBubble key={i} role={msg.role} text={msg.text} />
        ))}
        {loading && (
          <div className="flex items-center gap-2 text-gray-500 text-sm pl-11">
            <Loader2 size={16} className="animate-spin" />
            Thinking…
          </div>
        )}
        <div ref={scrollRef} />
      </div>

      {/* Quick questions */}
      <div className="flex flex-wrap gap-2 mb-3">
        {QUICK_QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => handleQuickQuestion(q)}
            className="px-3 py-1 rounded-full bg-gray-800/60 text-xs text-gray-400 hover:text-white hover:bg-gray-700 border border-gray-700/50 transition truncate max-w-xs"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="flex gap-3">
        <input
          className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 outline-none focus:ring-2 focus:ring-purple-500 transition"
          placeholder="Ask me anything about the CDU + VDU system…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
          disabled={loading}
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="px-4 rounded-xl bg-purple-600 hover:bg-purple-700 text-white transition disabled:opacity-50"
        >
          <Send size={18} />
        </button>
      </div>
    </div>
  );
}
