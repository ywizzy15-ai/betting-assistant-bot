"""
parser.py - Regex-based natural language command parser for betting requests.
Handles inputs like: "book me a 20 odd football", "5 draws with 65% probability"
"""
import re
from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)

# Mapping dictionaries for normalization
VALID_MARKETS = {
    "draw": "Draw",
    "over 2.5": "Over 2.5",
    "over 1.5": "Over 1.5",
    "btts": "BTTS",
    "both teams to score": "BTTS",
    "corners": "Corners",
    "match winner": "Match Winner",
    "home win": "Home Win",
    "away win": "Away Win",
}

VALID_SPORTS = {
    "football": "Football",
    "soccer": "Football",
    "basketball": "Basketball",
    "tennis": "Tennis",
    "cricket": "Cricket",
}

def parse_betting_command(text: str) -> Dict[str, Union[str, float, int]]:
    """
    Parse natural language betting request into structured parameters.
    Returns a dictionary with normalized sport, market, odds range, probability threshold, and count.
    """
    text = text.strip().lower()
    
    # Defaults
    params: Dict[str, Union[str, float, int]] = {
        "sport": "Football",
        "market": "Match Winner",
        "odds_min": 1.10,
        "odds_max": 10.00,
        "min_prob": 0.50,
        "count": 1,
        "raw_query": text
    }

    # 1. Parse Odds Target (e.g., "20 odd", "2.5 odds")
    odds_match = re.search(r'(\d+(?:\.\d+)?)\s*odd', text)
    if odds_match:
        target_odds = float(odds_match.group(1))
        # Apply heuristic buffer for search range
        params["odds_min"] = max(1.01, round(target_odds * 0.80, 2))
        params["odds_max"] = round(target_odds * 1.30, 2)

    # 2. Parse Probability Threshold (e.g., "65% probability", "70% chance")
    prob_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if prob_match:
        params["min_prob"] = round(float(prob_match.group(1)) / 100.0, 2)

    # 3. Parse Count/Quantity (e.g., "5 draws", "3 picks")
    count_match = re.search(r'(\d+)\s+(draw|bets|picks|tips|games|matches)', text)
    if count_match:
        params["count"] = int(count_match.group(1))
        # If count specifically mentions "draw", override market
        if count_match.group(2) == "draw" and params["market"] == "Match Winner":
            params["market"] = "Draw"

    # 4. Parse Market
    for key, val in VALID_MARKETS.items():
        if key in text:
            params["market"] = val
            # Auto-adjust odds ranges for common markets if not manually set by odds regex
            if "over" in key:
                params["odds_min"] = max(params["odds_min"], 1.30)
                params["odds_max"] = min(params["odds_max"], 3.50)
            elif key == "draw":
                params["odds_min"] = max(params["odds_min"], 2.50)
                params["odds_max"] = min(params["odds_max"], 5.00)
            elif key == "corners":
                params["odds_min"] = max(params["odds_min"], 1.50)
            break

    # 5. Parse Sport
    for key, val in VALID_SPORTS.items():
        if key in text:
            params["sport"] = val
            break

    logger.info(f"Parsed query: {text} -> {params}")
    return params

