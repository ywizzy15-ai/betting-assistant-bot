"""
groq_client.py - Async GROQ LLM integration for triple-layer divergence analysis.
Replaces Claude API with GROQ's ultra-fast inference. 
Provides async-safe execution, graceful fallbacks, and UI-optimized prompts.
"""
import asyncio
import logging
from typing import Dict, Optional
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS

logger = logging.getLogger(__name__)

class GroqAnalyzer:
    """
    Async-safe LLM wrapper for analyzing probability divergences 
    between Bookmaker, Polymarket, and ML models.
    """
    def __init__(self):
        # Initialize GROQ client if API key is present
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self._fallback = "💡 Divergence detected. ML and crowd models disagree with bookmaker odds."

    async def analyze_divergence(
        self,
        match_name: str,
        bk_probs: Dict[str, float],
        poly_probs: Dict[str, float],
        ml_probs: Dict[str, float],
        liquidity: float = 0.0,
        volume_24h: float = 0.0
    ) -> str:
        """
        Async-safe LLM analysis of triple-layer probability divergence.
        Returns a concise, UI-ready insight string.
        """
        if not self.client:
            logger.debug("GROQ API key missing. Returning heuristic fallback.")
            return self._fallback

        prompt = self._build_analysis_prompt(
            match_name, bk_probs, poly_probs, ml_probs, liquidity, volume_24h
        )

        try:
            # Run synchronous GROQ SDK in executor to prevent blocking aiogram event loop
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=GROQ_TEMPERATURE,
                    max_tokens=GROQ_MAX_TOKENS,
                )
            )
            insight = response.choices[0].message.content.strip()
            return insight if insight else self._fallback

        except Exception as e:
            logger.warning(f"GROQ analysis failed: {e}")
            return self._fallback

    def _build_analysis_prompt(
        self, match: str, bk: Dict, poly: Dict, ml: Dict, liquidity: float, vol: float
    ) -> str:
        """Formats the strict divergence analysis prompt aligned with the technical guide."""
        return (
            f"You are an expert sports betting analyst. Analyze the probability divergence for: {match}\n\n"
            f"| Source       | Home Win | Draw   | Away Win |\n"
            f"|--------------|----------|--------|----------|\n"
            f"| Bookmaker    | {bk.get('home',0):.1%} | {bk.get('draw',0):.1%} | {bk.get('away',0):.1%} |\n"
            f"| Polymarket   | {poly.get('home',0):.1%} | {poly.get('draw',0):.1%} | {poly.get('away',0):.1%} |\n"
            f"| ML Model     | {ml.get('home',0):.1%} | {ml.get('draw',0):.1%} | {ml.get('away',0):.1%} |\n\n"
            f"Market Metadata: Liquidity ${liquidity:,.0f} | 24h Vol ${vol:,.0f}\n\n"
            "INSTRUCTIONS:\n"
            "1. Identify the largest probability gap between sources.\n"
            "2. Explain briefly what this divergence indicates (e.g., market inefficiency, value edge, or noise).\n"
            "3. Return EXACTLY ONE concise sentence (max 15 words) suitable for a betting app UI.\n"
            "4. NO markdown, NO greetings, NO filler. Just the insight."
        )

# Singleton instance for global import
groq_analyzer = GroqAnalyzer()

