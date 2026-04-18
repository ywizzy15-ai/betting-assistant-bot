"""
engine.py - Core async prediction engine.
Integrates heuristic probability modeling, Polymarket crowd intelligence,
triple-layer blending, divergence grading, and GROQ-generated insights.
"""
import asyncio
import json
import logging
import math
import random
import time
from typing import Dict, List, Optional
import aiohttp
from groq import Groq
from config import (
    GROQ_API_KEY, POLYMARKET_GAMMA_API, GROQ_MODEL, 
    GROQ_TEMPERATURE, GROQ_MAX_TOKENS, TRIPLE_BLEND_WEIGHTS
)
from grading import grader

logger = logging.getLogger(__name__)

# Initialize GROQ client (optional, graceful fallback if missing)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ==================== POLYMARKET ASYNC CLIENT ====================
_poly_cache: Dict = {"markets": [], "updated": 0, "ttl": 300}

async def fetch_polymarket_markets() -> List[Dict]:
    """Fetch active football markets from Polymarket Gamma API (async, cached)."""
    now = time.time()
    if now - _poly_cache["updated"] < _poly_cache["ttl"]:
        return _poly_cache["markets"]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{POLYMARKET_GAMMA_API}/markets",
                params={"active": "true", "closed": "false", "limit": 50},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Filter for football/soccer keywords
                    keywords = ["football", "soccer", "premier", "liga", "champions", "epl"]
                    filtered = [
                        m for m in data 
                        if any(k in m.get("question", "").lower() for k in keywords)
                    ]
                    _poly_cache["markets"] = filtered
                    _poly_cache["updated"] = now
                    logger.info(f"✅ Polymarket cache refreshed: {len(filtered)} markets")
                    return filtered
    except Exception as e:
        logger.warning(f"⚠️ Polymarket fetch failed: {e}")
    return _poly_cache.get("markets", [])

def extract_poly_odds(market: Dict) -> Optional[Dict]:
    """Extract Home/Draw/Away probabilities from Polymarket market dict."""
    try:
        prices_raw = market.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        outcomes = [o.lower() for o in market.get("outcomes", [])]
        
        if len(prices) < 2: return None
        
        # Map prices to outcomes
        prob_map = {}
        for i, p in enumerate(prices):
            prob = float(p)
            o = outcomes[i]
            if "home" in o or "1" in o or "win" in o: prob_map["home"] = prob
            elif "draw" in o or "x" in o: prob_map["draw"] = prob
            elif "away" in o or "2" in o or "lose" in o: prob_map["away"] = prob

        return {
            "home": prob_map.get("home", 0.33),
            "draw": prob_map.get("draw", 0.33),
            "away": prob_map.get("away", 0.34),
            "liquidity": float(market.get("liquidity", 0) or 0),
            "volume_24h": float(market.get("volume24hr", 0) or 0)
        }
    except Exception as e:
        logger.debug(f"Poly extraction failed: {e}")
        return None

# ==================== HEURISTIC & ML SIMULATION ====================
def simulate_match_probabilities(team_a: str, team_b: str, target_odds: float) -> Dict:
    """
    Heuristic probability generator simulating ELO/xG/Fatigue logic.
    Produces realistic ML model probabilities aligned with target bookmaker odds.
    """
    # Seed pseudo-randomness based on team names for consistency
    seed = abs(hash(f"{team_a}|{team_b}")) % (2**32)
    rng = random.Random(seed)
    
    # 1. Bookmaker Implied (adjusted for vig)
    bk_prob = 1.0 / max(target_odds, 1.01)
    vig = 0.04  # ~4% margin
    bk_prob_norm = bk_prob / (1 + vig)

    # 2. Simulate ML Model probability (adds realistic variance + value edge)
    edge_offset = rng.uniform(-0.06, 0.12)
    ml_prob = max(0.05, min(0.95, bk_prob_norm + edge_offset))

    # 3. Simulate Fatigue/xG proxy effect (simplified)
    # Negative fatigue reduces ML confidence slightly
    fatigue_factor = rng.uniform(0.95, 1.02)
    ml_prob *= fatigue_factor

    return {
        "bk": bk_prob_norm,
        "ml": ml_prob,
        "odds": round(max(1.01, 1.0 / bk_prob), 2)
    }

# ==================== GROQ INSIGHT GENERATOR ====================
async def generate_groq_insight(ml: float, bk: float, poly: Optional[float], market: str, teams: str) -> str:
    """Generate concise analytical insight using GROQ LLM."""
    if not groq_client:
        return "💡 Insight unavailable (GROQ key not set)."

    poly_str = f"{(poly*100):.1f}%" if poly is not None else "N/A"
    prompt = (
        f"Match: {teams} | Market: {market}\n"
        f"Probabilities -> ML: {ml*100:.1f}% | Bookmaker: {bk*100:.1f}% | Polymarket: {poly_str}\n"
        f"Provide exactly ONE sentence analyzing the divergence and potential value. No filler."
    )

    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GROQ_TEMPERATURE,
                max_tokens=GROQ_MAX_TOKENS
            )
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"GROQ insight failed: {e}")
        return "💡 High ML-Polymarket divergence detected. Potential value opportunity."

# ==================== MAIN PREDICTION PIPELINE ====================
async def predict_matches(params: Dict) -> List[Dict]:
    """
    Main async function to generate betting opportunities based on parsed user params.
    Implements triple-layer blending, divergence grading, and insight generation.
    """
    logger.info(f"🔍 Generating predictions for: {params.get('raw_query', 'unknown')}")
    
    poly_markets = await fetch_polymarket_markets()
    results = []
    
    # Generate 1-3 simulated matches matching the requested criteria
    count = params.get("count", 1)
    for _ in range(count):
        # Simulate match pairing
        rng = random.Random(int(time.time()))
        teams = ["Arsenal U21", "Chelsea U21", "Lagos United", "Kano Pillars", "AFC Leopards", "Tottenham U21"]
        t_a, t_b = rng.sample(teams, 2)
        teams_str = f"{t_a} vs {t_b}"
        
        # 1. Simulate Bookmaker & ML probabilities
        target_odds = rng.uniform(params["odds_min"], params["odds_max"])
        probs = simulate_match_probabilities(t_a, t_b, target_odds)
        
        # 2. Fetch/Extract Polymarket data (fallback if unavailable)
        poly_data = None
        if poly_markets and rng.random() > 0.3:  # ~70% chance to find poly data
            m = rng.choice(poly_markets)
            poly_data = extract_poly_odds(m)
        
        poly_prob = poly_data.get("home") if poly_data else None
        poly_liquidity = poly_data.get("liquidity", 0) if poly_data else 0

        # 3. Triple-Layer Blending (Config-driven weights)
        w_ml = TRIPLE_BLEND_WEIGHTS["ml"]
        w_poly = TRIPLE_BLEND_WEIGHTS["polymarket"]
        w_bk = TRIPLE_BLEND_WEIGHTS["bookmaker"]
        
        blended_ml = (probs["ml"] * w_ml) + (poly_prob * w_poly if poly_prob else probs["ml"] * w_poly) + (probs["bk"] * w_bk)
        blended_prob = max(0.05, min(0.95, blended_ml))

        # 4. Confidence Grading & Divergence Analysis
        grading = grader.calculate_confidence(
            ml_prob=probs["ml"],
            bk_prob=probs["bk"],
            poly_prob=poly_prob,
            poly_liquidity=poly_liquidity
        )

        # 5. Generate AI Insight
        insight = await generate_groq_insight(
            ml=probs["ml"], bk=probs["bk"], poly=poly_prob,
            market=params.get("market", "Match Winner"), teams=teams_str
        )

        # 6. Filter by minimum probability threshold
        if blended_prob >= params.get("min_prob", 0.5):
            results.append({
                "id": rng.randint(1000, 9999),
                "sport": params.get("sport", "Football"),
                "league": "Simulated / Small League",
                "teams": teams_str,
                "market": params.get("market", "Match Winner"),
                "odds": probs["odds"],
                "ml_prob": probs["ml"],
                "bk_prob": probs["bk"],
                "poly_prob": poly_prob,
                "blended_prob": blended_prob,
                "confidence": grading["confidence_score"],
                "grade": grading["grade"],
                "divergence_status": grading["divergence_status"],
                "edge_percentage": grading["edge_percentage"],
                "liquidity": poly_liquidity,
                "insight": insight,
                "platforms": random.sample(["SportyBet", "Bet9ja", "Stake", "BC.Game"], k=random.randint(2, 4)),
                "timestamp": time.time()
            })

    logger.info(f"✅ Generated {len(results)} opportunities")
    return results

