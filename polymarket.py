"""
polymarket.py - Async Polymarket Gamma API client.
Fetches football prediction markets, extracts crowd-sourced probabilities,
and caches results to respect API rate limits.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import aiohttp
from config import POLYMARKET_GAMMA_API

logger = logging.getLogger(__name__)

@dataclass
class PolymarketOdds:
    """Structured container for extracted Polymarket probabilities."""
    home_win: float
    draw: Optional[float]
    away_win: float
    liquidity: float
    volume_24h: float
    market_slug: str
    question: str
    last_updated: str

class PolymarketClient:
    """
    Async client for Polymarket Gamma API.
    Public, no-auth REST endpoint. Filters for football/soccer markets.
    """
    def __init__(self, ttl: int = 300):
        self.base_url = POLYMARKET_GAMMA_API
        self.cache: List[Dict] = []
        self.cache_time: float = 0.0
        self.ttl = ttl
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
        }
        # Keywords to filter football/soccer markets
        self.keywords = [
            "soccer", "football", "premier league", "la liga", "bundesliga",
            "serie a", "ligue 1", "champions league", "uefa", "epl",
            "manchester", "liverpool", "arsenal", "chelsea", "real madrid"
        ]

    async def fetch_markets(self, limit: int = 50) -> List[Dict]:
        """Fetch active markets with TTL caching and polite rate limiting."""
        now = time.time()
        if self.cache and (now - self.cache_time) < self.ttl:
            return self.cache

        all_markets: List[Dict] = []
        offset = 0
        timeout = aiohttp.ClientTimeout(total=15)

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                while offset < limit:
                    async with session.get(
                        f"{self.base_url}/markets",
                        params={"active": "true", "closed": "false", "limit": 50, "offset": offset},
                        timeout=timeout
                    ) as resp:
                        if resp.status != 200:
                            logger.warning(f"Polymarket API returned HTTP {resp.status}")
                            break
                        
                        markets = await resp.json()
                        if not markets:
                            break

                        for m in markets:
                            question = m.get("question", "").lower()
                            desc = m.get("description", "").lower()
                            if any(kw in question or kw in desc for kw in self.keywords):
                                all_markets.append(m)
                        
                        offset += 50
                        await asyncio.sleep(0.5)  # Polite rate limit (~2 req/sec)
        except Exception as e:
            logger.error(f"Polymarket fetch failed: {e}")

        self.cache = all_markets
        self.cache_time = time.time()
        logger.info(f"✅ Polymarket cache refreshed: {len(all_markets)} markets")
        return self.cache

    def extract_odds(self, market: Dict) -> Optional[PolymarketOdds]:
        """
        Extract probabilities from market data.
        Handles binary (2-way) and standard 3-way (H/D/A) markets.
        """
        try:
            outcomes = market.get("outcomes", [])
            prices_raw = market.get("outcomePrices", "[]")
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw

            if len(prices) < 2:
                return None

            prices = [float(p) for p in prices]
            outcomes_lower = [o.lower() for o in outcomes]

            # Binary market (e.g., "Team A to win?")
            if len(prices) == 2:
                return PolymarketOdds(
                    home_win=prices[0],
                    draw=None,
                    away_win=prices[1],
                    liquidity=float(market.get("liquidity", 0) or 0),
                    volume_24h=float(market.get("volume24hr", 0) or 0),
                    market_slug=market.get("slug", ""),
                    question=market.get("question", ""),
                    last_updated=market.get("updatedAt", "")
                )

            # 3-way market (Home / Draw / Away)
            if len(prices) >= 3:
                home_idx = next((i for i, o in enumerate(outcomes_lower) if any(x in o for x in ["home", "win", "1"])), 0)
                draw_idx = next((i for i, o in enumerate(outcomes_lower) if any(x in o for x in ["draw", "tie", "x"])), 1)
                away_idx = next((i for i, o in enumerate(outcomes_lower) if any(x in o for x in ["away", "lose", "2"])), 2)

                return PolymarketOdds(
                    home_win=prices[home_idx],
                    draw=prices[draw_idx],
                    away_win=prices[away_idx],
                    liquidity=float(market.get("liquidity", 0) or 0),
                    volume_24h=float(market.get("volume24hr", 0) or 0),
                    market_slug=market.get("slug", ""),
                    question=market.get("question", ""),
                    last_updated=market.get("updatedAt", "")
                )
        except Exception as e:
            logger.debug(f"Polymarket extraction failed: {e}")
            return None

    async def get_football_markets(self, limit: int = 50) -> List[PolymarketOdds]:
        """Fetch and parse only valid football market odds."""
        raw_markets = await self.fetch_markets(limit)
        parsed = [self.extract_odds(m) for m in raw_markets]
        return [odds for odds in parsed if odds is not None]

# Singleton instance for global import
poly_client = PolymarketClient()

