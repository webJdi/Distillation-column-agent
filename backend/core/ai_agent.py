"""
AI Agent — the "brain" that explains the system, generates reports,
and answers user questions about the CDU+VDU optimization.

Uses Google's Gemini models with Chain-of-Thought (CoT) reasoning,
specializing in chemical process engineering of Crude Distillation Units
and report creation.
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional
from loguru import logger

from backend.config import settings

# Google Generative AI SDK
try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore


SYSTEM_PROMPT = """You are a specialist chemical process engineer and AI assistant for a Crude Distillation Unit (CDU) and Vacuum Distillation Unit (VDU) optimization system. You think step-by-step using Chain-of-Thought reasoning before giving final answers.

**Your areas of expertise:**
- Petroleum refining: atmospheric and vacuum distillation, TBP/ASTM cut-point analysis
- Process simulation: DWSIM-based flowsheet modelling, Peng-Robinson EOS
- Reinforcement learning: SAC, TD3, PPO applied to continuous process control
- Techno-economic analysis: product pricing, energy costs, profit optimization

**System overview:**
The simulation uses main_sim.dwxmz with TWO columns:
  1. **Atmospheric Column (ADU)** — receives preheated crude and produces:
     Uncondensed Gas, USN (Unstabilized Naphtha ≈ Naphtha+LPG), Heavy Naphtha (HN),
     SKO (Jet Fuel/Kerosene), Light Diesel (LD), Heavy Diesel (HD)
  2. **Vacuum Column (VDU)** — receives the atmospheric residue and produces:
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

Always be precise with numbers, reference the actual data provided, and explain complex concepts in practical terms. Use proper engineering units (°C, kPa, kg/h, $/bbl)."""


class AIAgent:
    """AI agent using Google Gemini with Chain-of-Thought reasoning
    for CDU+VDU process engineering analysis and report creation."""

    def __init__(self):
        self.model = None
        if settings.GEMINI_API_KEY and genai:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                system_instruction=SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=4096,
                ),
            )
            self._chat = self.model.start_chat(history=[])
            logger.info(f"AI Agent initialized with Google Gemini ({settings.GEMINI_MODEL})")
        else:
            self._chat = None
            logger.warning("AI Agent running in offline mode (no Gemini API key)")

        self.conversation_history: list[dict] = []

    async def ask(
        self,
        question: str,
        context: Optional[dict] = None,
        include_state: bool = True,
    ) -> dict:
        """
        Answer a question about the CDU+VDU system using CoT reasoning.
        """
        # Build context message
        context_parts = []
        if context:
            if "prices" in context:
                context_parts.append(f"Current product prices: {json.dumps(context['prices'], indent=2)}")
            if "state" in context and include_state:
                context_parts.append(f"Current column state: {json.dumps(context['state'], indent=2)}")
            if "training_progress" in context:
                context_parts.append(f"Training progress: {json.dumps(context['training_progress'], indent=2)}")
            if "action" in context:
                context_parts.append(f"Last agent action: {json.dumps(context['action'], indent=2)}")

        context_msg = "\n\n".join(context_parts) if context_parts else "No current system data available."

        # Wrap with CoT instruction
        full_prompt = (
            f"**System Context:**\n{context_msg}\n\n"
            f"**Question:** {question}\n\n"
            "Think step-by-step through the relevant process engineering principles, "
            "then provide your answer."
        )

        if self.model and self._chat:
            try:
                response = self._chat.send_message(full_prompt)
                answer = response.text

                # Update conversation history for offline fallback
                self.conversation_history.append({"role": "user", "content": question})
                self.conversation_history.append({"role": "assistant", "content": answer})

                return {
                    "answer": answer,
                    "sources": ["DWSIM simulation", "RL agent observations", "Gemini CoT analysis"],
                    "suggested_actions": self._extract_suggestions(answer),
                }
            except Exception as exc:
                logger.error(f"Gemini API error: {exc}")
                return self._offline_response(question, context)
        else:
            return self._offline_response(question, context)

    async def generate_report(
        self,
        report_type: str = "summary",
        data: Optional[dict] = None,
    ) -> dict:
        """
        Generate a structured report about the CDU optimization.
        
        Args:
            report_type: 'summary', 'detailed', 'optimization', 'comparison'
            data: Data to include in the report
            
        Returns:
            dict with 'report_id', 'content', 'summary'
        """
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
                "agent's recommended settings for both ADU and VDU. Show potential profit improvement "
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

        # Save report to file
        report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Report", "generated")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"{report_id}.md")

        report_content = f"""# CDU Optimization Report — {report_type.title()}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Type:** {report_type}

---

{result['answer']}

---
*Generated by CDU Optimizer AI Agent*
"""
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

    def _offline_response(self, question: str, context: Optional[dict]) -> dict:
        """Provide a rule-based response when OpenAI is unavailable."""
        q_lower = question.lower()

        if any(w in q_lower for w in ["reward", "profit", "revenue"]):
            answer = self._explain_reward(context)
        elif any(w in q_lower for w in ["action", "reflux", "temperature", "steam"]):
            answer = self._explain_actions(context)
        elif any(w in q_lower for w in ["train", "learn", "episode"]):
            answer = self._explain_training(context)
        elif any(w in q_lower for w in ["product", "yield", "flow"]):
            answer = self._explain_products(context)
        elif any(w in q_lower for w in ["disturb", "feed", "crude"]):
            answer = self._explain_disturbances(context)
        elif any(w in q_lower for w in ["safe", "limit", "constraint"]):
            answer = self._explain_safety(context)
        else:
            answer = (
                "The CDU+VDU Optimizer uses reinforcement learning agents (SAC, TD3, PPO) "
                "to learn optimal column operating parameters. The agent controls reflux ratios, "
                "side-stream draw temperatures, and stripping steam rates on both the "
                "atmospheric and vacuum columns to maximize profit "
                "(product revenue minus energy costs) while respecting safety constraints.\n\n"
                "**Atmospheric products:** Uncondensed Gas, USN (Naphtha), HN, SKO (Jet Fuel), LD, HD\n"
                "**Vacuum products:** Vac Diesel, VGO, Slop Cut, Vac Residue"
            )

        return {
            "answer": answer,
            "sources": ["Built-in knowledge base"],
            "suggested_actions": [],
        }

    def _explain_reward(self, ctx: Optional[dict]) -> str:
        text = (
            "**Reward Function:**\n"
            "The RL agent's reward is calculated as:\n\n"
            "```\nReward = Σ(product_flow × product_price) − energy_cost − safety_penalty\n```\n\n"
            "- **Revenue:** Each of the 10 product stream's mass flow rate (kg/h) multiplied by its market price\n"
            "- **Energy cost:** (ADU condenser + reboiler + VDU condenser + reboiler duty) × $0.05/kWh\n"
            "- **Reflux penalty:** Discourages unnecessarily high reflux ratios on both columns\n"
            "- **Safety penalty:** Applied when temperatures approach operational limits\n"
        )
        if ctx and "prices" in ctx:
            text += f"\nCurrent prices: {json.dumps(ctx['prices'], indent=2)}"
        return text

    def _explain_actions(self, ctx: Optional[dict]) -> str:
        return (
            "**Agent Actions (10 Control Variables):**\n\n"
            "**Atmospheric Column (7):**\n"
            "1. **Reflux ratio** (0.5–8.0): Controls ADU overhead separation quality\n"
            "2. **USN draw temperature** (60–180°C): Naphtha (+ LPG) cut point\n"
            "3. **HN draw temperature** (120–220°C): Heavy naphtha cut point\n"
            "4. **SKO draw temperature** (170–280°C): Jet fuel/kerosene cut point\n"
            "5. **LD draw temperature** (240–340°C): Light diesel cut point\n"
            "6. **HD draw temperature** (300–380°C): Heavy diesel cut point\n"
            "7. **Atmos steam rate** (0–5000 kg/h): Stripping steam in the ADU\n\n"
            "**Vacuum Column (3):**\n"
            "8. **Vac reflux ratio** (0.5–5.0): Controls VDU overhead separation\n"
            "9. **Vac Diesel draw temperature** (200–350°C): Vacuum diesel cut point\n"
            "10. **VGO draw temperature** (300–420°C): Vacuum gas oil cut point\n\n"
            "Higher draw temperatures shift product to heavier fractions. The agent learns to "
            "balance all 10 operating parameters across both columns to maximize total profit."
        )

    def _explain_training(self, ctx: Optional[dict]) -> str:
        return (
            "**Training Process:**\n"
            "The agent supports three architectures: **SAC** (default), **TD3**, and **PPO**.\n\n"
            "- **Curriculum learning:** Training starts with small disturbances (easy) and "
            "progressively increases difficulty\n"
            "- **Episodes:** Each episode runs up to 200 steps of column adjustment\n"
            "- **Action space:** 10-dimensional continuous (7 ADU + 3 VDU controls)\n"
            "- **Observation space:** 31-dimensional (10 flows + 10 temps + 11 column state vars)\n"
            "- **Replay buffer:** Stores past experiences for efficient off-policy learning (SAC/TD3)\n"
        )

    def _explain_products(self, ctx: Optional[dict]) -> str:
        return (
            "**Product Streams (ADU + VDU):**\n\n"
            "| Product | Source | Typical Cut Range | Key Use |\n"
            "|---------|--------|------------------|---------|\n"
            "| Uncondensed Gas | ADU overhead | C1–C4 | Fuel gas |\n"
            "| USN (Naphtha) | ADU | IBP–100°C | Gasoline blending, petrochemicals |\n"
            "| HN | ADU | 100–160°C | Reformer feed |\n"
            "| SKO (Jet Fuel) | ADU | 160–240°C | Aviation fuel, kerosene |\n"
            "| LD (Light Diesel) | ADU | 240–300°C | Diesel blending |\n"
            "| HD (Heavy Diesel) | ADU | 300–370°C | Diesel, FCC feed |\n"
            "| Vac Diesel | VDU | 370–420°C | Diesel blending |\n"
            "| VGO | VDU | 420–500°C | FCC / hydrocracker feed |\n"
            "| Slop Cut | VDU | transition | Recycle / reprocessing |\n"
            "| Vac Residue | VDU bottom | 500°C+ | Fuel oil, coker feed, bitumen |\n"
        )

    def _explain_disturbances(self, ctx: Optional[dict]) -> str:
        return (
            "**Feed Disturbances:**\n"
            "You can introduce disturbances to test the agent's robustness:\n\n"
            "- **Feed temperature** (±50°C): Simulates furnace performance changes\n"
            "- **Feed pressure** (±50 kPa): Simulates upstream pressure variations\n"
            "- **Feed flow rate** (±30%): Simulates throughput changes\n"
            "- **API gravity** (±10): Simulates crude quality changes (lighter vs heavier crude)\n"
            "- **Crude blend:** Switch between different crude oil types "
            "(WTI Light, Azeri Light, Tapis, etc.)\n"
        )

    def _explain_safety(self, ctx: Optional[dict]) -> str:
        return (
            "**Safety Constraints:**\n"
            f"- Max column temperature: {settings.MAX_COLUMN_TEMP}°C\n"
            f"- Min column temperature: {settings.MIN_COLUMN_TEMP}°C\n"
            f"- Max column pressure: {settings.MAX_COLUMN_PRESSURE} kPa\n"
            f"- Min column pressure: {settings.MIN_COLUMN_PRESSURE} kPa\n\n"
            "The agent receives increasing penalties as it approaches these limits, "
            "and the episode terminates if limits are exceeded."
        )

    def _extract_suggestions(self, text: str) -> list[str]:
        """Extract actionable suggestions from the AI response."""
        suggestions = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith(("- **Recommendation", "- **Action", "- **Suggest", "→")):
                suggestions.append(line.lstrip("- →*").strip())
        return suggestions[:5]

    def clear_history(self) -> None:
        """Reset conversation history and restart Gemini chat session."""
        self.conversation_history = []
        if self.model:
            self._chat = self.model.start_chat(history=[])
