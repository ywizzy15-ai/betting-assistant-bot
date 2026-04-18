"""
prompt.py - LLM prompt templates optimized for GROQ (Llama 3.3 70B).
Replaces Claude templates from the technical guide with strict, structured instructions
designed for low-latency inference and reliable parsing.
"""

class PromptTemplates:
    """
    Centralized prompt templates for triple-layer analysis and contextual feature generation.
    All templates use strict formatting constraints to prevent markdown/JSON parsing failures.
    """

    # 1. Divergence Analysis (Core UI Insight - Max 15 words)
    DIVERGENCE_ANALYSIS = """You are an expert sports betting analyst. Analyze the probability divergence for: {match_name}

| Source       | Home Win | Draw   | Away Win |
|--------------|----------|--------|----------|
| Bookmaker    | {bk_home:.1%} | {bk_draw:.1%} | {bk_away:.1%} |
| Polymarket   | {poly_home:.1%} | {poly_draw:.1%} | {poly_away:.1%} |
| ML Model     | {ml_home:.1%} | {ml_draw:.1%} | {ml_away:.1%} |

Market Meta Liquidity ${liquidity:,.0f} | 24h Vol ${volume_24h:,.0f}

INSTRUCTIONS:
1. Identify the largest probability gap between sources.
2. Explain briefly what this divergence indicates (e.g., market inefficiency, value edge, or noise).
3. Return EXACTLY ONE concise sentence (max 15 words) suitable for a betting app UI.
4. NO markdown, NO greetings, NO filler. Just the insight."""

    # 2. Contextual Feature Generation (Strict JSON)
    CONTEXTUAL_FEATURES = """You are an expert football match analyst. Analyze the upcoming match and return ONLY JSON (no markdown, no comments, no backticks) with the following scores on a scale from 0.0 to 1.0:

Match: {home_team} (home) vs {away_team} (away)
League: {league}

{home_team} stats over last 5 matches:
- Avg goals scored: {home_gf:.2f}
- Avg goals conceded: {home_ga:.2f}
- Avg shots: {home_shots:.1f}
- Avg shots on target: {home_sot:.1f}
- Form (avg points): {home_form:.2f}

{away_team} stats over last 5 matches:
- Avg goals scored: {away_gf:.2f}
- Avg goals conceded: {away_ga:.2f}
- Avg shots: {away_shots:.1f}
- Avg shots on target: {away_sot:.1f}
- Form (avg points): {away_form:.2f}

Return JSON strictly in this format:
{{
    "home_attack_strength": <float>,
    "home_defense_strength": <float>,
    "away_attack_strength": <float>,
    "away_defense_strength": <float>,
    "home_momentum": <float>,
    "away_momentum": <float>,
    "match_intensity_prediction": <float>,
    "upset_probability": <float>,
    "home_win_confidence": <float>,
    "draw_likelihood": <float>,
    "reasoning": "<brief 1-2 sentence explanation>"
}}"""

    # 3. Detailed Match Report (Max 150 words)
    MATCH_REPORT = """You are a professional football analyst. Based on the machine learning model data and team statistics, write a concise but insightful analytical report on the upcoming match.

## Model Data
Match: **{home_team}** vs **{away_team}** ({league})

Model probabilities (ML Ensemble):
- {home_team} win: {prob_home:.1%}
- Draw: {prob_draw:.1%}
- {away_team} win: {prob_away:.1%}

{home_team} stats (last 5 matches):
- Goals scored (avg): {home_gf:.2f}
- Goals conceded (avg): {home_ga:.2f}
- Shots on target (avg): {home_sot:.1f}
- Form (avg points): {home_form:.2f}

{away_team} stats (last 5 matches):
- Goals scored (avg): {away_gf:.2f}
- Goals conceded (avg): {away_ga:.2f}
- Shots on target (avg): {away_sot:.1f}
- Form (avg points): {away_form:.2f}

## Task
Write an analytical report that includes:
1. Key factors affecting the prediction
2. Strengths and weaknesses of each team
3. Most likely outcome prediction
4. Confidence level (high / medium / low)
5. Potential risks and upset scenarios

Write concisely, professionally, no filler. Max 150 words."""

    # 4. Batch Matchday Analysis
    MATCHDAY_BATCH = """Analyze the upcoming matchday. For each match, provide:
- Prediction (1X2)
- Confidence (⭐ low, ⭐⭐ medium, ⭐⭐⭐ high)
- Brief comment (1 sentence)

Matches:
{matches_list}

Return in table format. At the end, add the 1-2 best picks of the matchday (highest confidence)."""

    # ==================== FORMATTER HELPERS ====================
    @classmethod
    def format_divergence(cls, **kwargs) -> str:
        """Format divergence analysis prompt with probability & liquidity data."""
        return cls.DIVERGENCE_ANALYSIS.format(**kwargs)

    @classmethod
    def format_context(cls, **kwargs) -> str:
        """Format contextual feature prompt with rolling team stats."""
        return cls.CONTEXTUAL_FEATURES.format(**kwargs)

    @classmethod
    def format_report(cls, **kwargs) -> str:
        """Format detailed match report prompt."""
        return cls.MATCH_REPORT.format(**kwargs)

    @classmethod
    def format_batch(cls, **kwargs) -> str:
        """Format batch matchday analysis prompt."""
        return cls.MATCHDAY_BATCH.format(**kwargs)

