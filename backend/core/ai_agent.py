"""
AI Agent -- the "brain" that explains the system, generates reports,
and answers user questions about the CDU+VDU optimization.

Uses Nvidia Nemotron 3 Nano (30B-A3B) via OpenRouter with Chain-of-Thought
(CoT) reasoning, specializing in chemical process engineering of Crude
Distillation Units and Vacuum Distillation Units.
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional
from loguru import logger

import httpx

from backend.config import settings


SYSTEM_PROMPT = """You are a specialist chemical process engineer and AI assistant for a Crude Distillation Unit (CDU) and Vacuum Distillation Unit (VDU) optimization system. You think step-by-step using Chain-of-Thought reasoning before giving final answers.

**Your areas of expertise:**
- Petroleum refining: atmospheric and vacuum distillation, TBP/ASTM cut-point analysis
- Process simulation: DWSIM-based flowsheet modelling, Peng-Robinson EOS
- Reinforcement learning: SAC, TD3, PPO applied to continuous process control
- Techno-economic analysis: product pricing, energy costs, profit optimization

**System overview:**
The simulation uses main_sim.dwxmz with TWO columns:
  1. **Atmospheric Column (ADU)** -- receives preheated crude and produces:
     Uncondensed Gas, USN (Unstabilized Naphtha ~= Naphtha+LPG), Heavy Naphtha (HN),
     SKO (Jet Fuel/Kerosene), Light Diesel (LD), Heavy Diesel (HD)
  2. **Vacuum Column (VDU)** -- receives the atmospheric residue and produces:
     Vacuum Diesel, Vacuum Gas Oil (VGO), Slop Cut, Vacuum Residue

**Agent actions (10 dimensions):**
ADU: reflux_ratio, usn_draw_temp, hn_draw_temp, sko_draw_temp, ld_draw_temp, hd_draw_temp, atmos_steam_rate
VDU: vac_reflux_ratio, vac_diesel_draw_temp, vgo_draw_temp

**Thinking approach:**
When answering ANY question, ALWAYS follow this internal reasoning pattern:
1. Identify what process engineering principles are relevant
2. Consider the thermodynamic and mass-balance implications
3. Evaluate economic trade-offs (product value vs energy cost)
4. Formulate a clear, data-backed recommendation

**Report creation:**
When generating reports, structure them professionally with:
- Executive summary, methodology, findings, recommendations
- Tables with actual data values from the system
- Clear engineering reasoning for each recommendation

Always be precise with numbers, reference the actual data provided, and explain complex concepts in practical terms. Use proper engineering units (C, kPa, kg/h, $/bbl)."""


class AIAgent:
    """AI agent using Nvidia Nemotron 3 Nano via OpenRouter with CoT reasoning
    for CDU+VDU process engineering analysis and report creation."""

    def __init__(self):
        self._api_key = settings.OPENROUTER_API_KEY
        self._model = settings.OPENROUTER_MODEL
        self._base_url = settings.OPENROUTER_BASE
        self.conversation_history: list[dict] = []

        if self._api_key:
            logger.info(f"AI Agent initialized with Nvidia Nemotron ({self._model}) via OpenRouter")
        else:
            logger.warning("AI Agent running in offline mode (no OPENROUTER_API_KEY set)")

    # -- Internal helpers ---------------------------------------------------

    def _build_messages(self, question: str, context: Optional[dict]) -> list[dict]:
        """Build the OpenAI-compatible messages list including conversation history."""
        context_parts = []
        if context:
            if "prices" in context:
                context_parts.append(
                    f"Current product prices: {json.dumps(context['prices'], indent=2)}"
                )
            if "state" in context:
                context_parts.append(
                    f"Current column state: {json.dumps(context['state'], indent=2)}"
                )
            if "training_progress" in context:
                context_parts.append(
                    f"Training progress: {json.dumps(context['training_progress'], indent=2)}"
                )
            if "action" in context:
                context_parts.append(
                    f"Last agent action: {json.dumps(context['action'], indent=2)}"
                )

        context_msg = "\n\n".join(context_parts) if context_parts else "No current system data available."
        user_content = (
            f"**System Context:**\n{context_msg}\n\n"
            f"**Question:** {question}\n\n"
            "Think step-by-step through the relevant process engineering principles, "
            "then provide your answer."
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.conversation_history[-10:])
        messages.append({"role": "user", "content": user_content})
        return messages

    async def _call_openrouter(self, messages: list[dict]) -> str:
        """Make an async HTTP request to OpenRouter and return the response text."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Distillation-column-agent",
            "X-Title": "CDU Optimizer AI Agent",
        }
        payload = {"model": self._model, "messages": messages, "stream": False}
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(self._base_url, headers=headers, json=payload)
            resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # -- Public interface ---------------------------------------------------

    async def ask(
        self,
        question: str,
        context: Optional[dict] = None,
        include_state: bool = True,
    ) -> dict:
        """Answer a question about the CDU+VDU system using CoT reasoning."""
        if not include_state and context:
            context = {k: v for k, v in context.items() if k != "state"}

        if self._api_key:
            try:
                messages = self._build_messages(question, context)
                answer = await self._call_openrouter(messages)
                self.conversation_history.append({"role": "user", "content": question})
                self.conversation_history.append({"role": "assistant", "content": answer})
                return {
                    "answer": answer,
                    "sources": ["DWSIM simulation", "RL agent observations", "Nemotron CoT analysis"],
                    "suggested_actions": self._extract_suggestions(answer),
                }
            except Exception as exc:
                logger.error(f"OpenRouter/Nemotron API error: {exc}")
                return self._offline_response(question, context)
        else:
            return self._offline_response(question, context)

    async def generate_report(
        self,
        report_type: str = "summary",
        data: Optional[dict] = None,
    ) -> dict:
        """Generate a structured report about the CDU optimization."""
        report_prompts = {
            "summary": (
                "Generate a concise executive summary of the current CDU+VDU optimization status. "
                "Cover: atmospheric column (Uncondensed Gas, USN, HN, SKO, LD, HD) and vacuum column "
                "(Vac Diesel, VGO, Slop Cut, Vac Residue). Include key metrics, product yields, "
                "profitability, energy consumption, and any alerts."
            ),
            "detailed": (
                "Generate a detailed technical process engineering report covering both the atmospheric "
                "and vacuum columns. Include: operating parameters for both columns, product quality/"
                "quantity analysis for all 10 streams, energy consumption breakdown, safety margins, "
                "and optimization recommendations with engineering justification."
            ),
            "optimization": (
                "Generate an optimization report comparing current operating conditions to the RL "
                "agent recommended settings for both ADU and VDU. Show potential profit improvement "
                "per product stream and explain trade-offs between product yields."
            ),
            "comparison": (
                "Generate a comparative analysis of different operating scenarios across both columns. "
                "Highlight which scenario maximizes total profit and under what market conditions "
                "each is optimal."
            ),
        }

        prompt = report_prompts.get(report_type, report_prompts["summary"])
        if data:
            prompt += f"\n\nData for the report:\n{json.dumps(data, indent=2, default=str)}"

        result = await self.ask(prompt, context=data, include_state=True)
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{report_type}"
        report_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Report", "generated",
        )
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"{report_id}.md")
        report_content = (
            f"# CDU Optimization Report -- {report_type.title()}\n"
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Type:** {report_type}\n\n---\n\n{result['answer']}\n\n---\n"
            "*Generated by CDU Optimizer AI Agent (Nvidia Nemotron 3 Nano via OpenRouter)*\n"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        return {
            "report_id": report_id,
            "report_type": report_type,
            "file_path": report_path,
            "content": report_content,
            "summary": result["answer"][:300] + "..." if len(result["answer"]) > 300 else result["answer"],
            "created_at": datetime.now().isoformat(),
        }

    # -- Offline fallback helpers ------------------------------------------

    def _offline_response(self, question: str, context: Optional[dict]) -> dict:
        """Provide a rule-based response when the LLM is unavailable."""
        q = question.lower()
        if any(w in q for w in ["reward", "profit", "revenue"]):
            answer = self._explain_reward(context)
        elif any(w in q for w in ["action", "reflux", "temperature", "steam"]):
            answer = self._explain_actions(context)
        elif any(w in q for w in ["train", "learn", "episode"]):
            answer = self._explain_training(context)
        elif any(w in q for w in ["product", "yield", "flow"]):
            answer = self._explain_products(context)
        elif any(w in q for w in ["disturb", "feed", "crude"]):
            answer = self._explain_disturbances(context)
        elif any(w in q for w in ["safe", "limit", "constraint"]):
            answer = self._explain_safety(context)
        else:
            answer = (
                "The CDU+VDU Optimizer uses SAC/TD3/PPO reinforcement learning to optimize "
                "10 control variables across both the atmospheric and vacuum columns.\n\n"
                "**Atmospheric products:** Uncondensed Gas, USN, HN, SKO, LD, HD\n"
                "**Vacuum products:** Vac Diesel, VGO, Slop Cut, Vac Residue\n\n"
                "*(Set OPENROUTER_API_KEY to enable live Nvidia Nemotron 3 Nano responses.)*"
            )
        return {"answer": answer, "sources": ["Built-in knowledge base"], "suggested_actions": []}

    def _explain_reward(self, ctx: Optional[dict]) -> str:
        text = (
            "**Reward = sum(flow * price) - energy_cost - safety_penalty**\n\n"
            "Revenue from all 10 product streams minus ADU+VDU energy costs."
        )
        if ctx and "prices" in ctx:
            text += f"\nCurrent prices: {json.dumps(ctx['prices'], indent=2)}"
        return text

    def _explain_actions(self, _ctx: Optional[dict]) -> str:
        return (
            "**10 Control Variables:**\n"
            "ADU (7): reflux_ratio, usn_draw_temp, hn_draw_temp, sko_draw_temp, "
            "ld_draw_temp, hd_draw_temp, atmos_steam_rate\n"
            "VDU (3): vac_reflux_ratio, vac_diesel_draw_temp, vgo_draw_temp"
        )

    def _explain_training(self, _ctx: Optional[dict]) -> str:
        return (
            "Supports SAC (default), TD3, PPO. Episodes up to 200 steps. "
            "31-dim observations, 10-dim continuous actions."
        )

    def _explain_products(self, _ctx: Optional[dict]) -> str:
        return (
            "ADU: Uncondensed Gas, USN, HN, SKO, LD, HD\n"
            "VDU: Vac Diesel, VGO, Slop Cut, Vac Residue"
        )

    def _explain_disturbances(self, _ctx: Optional[dict]) -> str:
        return "Disturbances: feed temperature, pressure, flow rate, API gravity, crude blend."

    def _explain_safety(self, _ctx: Optional[dict]) -> str:
        return (
            f"Max temp: {settings.MAX_COLUMN_TEMP}C | Min temp: {settings.MIN_COLUMN_TEMP}C\n"
            f"Max pressure: {settings.MAX_COLUMN_PRESSURE} kPa | "
            f"Min pressure: {settings.MIN_COLUMN_PRESSURE} kPa"
        )

    def _extract_suggestions(self, text: str) -> list[str]:
        suggestions = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith(("- **Recommendation", "- **Action", "- **Suggest", "->")):
                suggestions.append(line.lstrip("- ->*").strip())
        return suggestions[:5]

    def clear_history(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []
