"""
divergence.py - Triple-layer probability divergence analyzer.
Combines Bookmaker, Polymarket, and ML probabilities to compute 
KL-divergence, absolute gaps, agreement flags, and liquidity features.
"""
import numpy as np
import logging
from typing import Dict, Optional
from config import TRIPLE_BLEND_WEIGHTS

logger = logging.getLogger(__name__)

class DivergenceAnalyzer:
    """
    Stateless analyzer for computing probability divergences and blended features.
    Designed for integration into the prediction pipeline.
    """

    @staticmethod
    def _safe_divide_log(p: float, q: float, epsilon: float = 1e-6) -> float:
        """Safe computation of p * log(p/q) to prevent domain errors."""
        p = max(p, epsilon)
        q = max(q, epsilon)
        return p * np.log(p / q)

    @classmethod
    def compute_kl_divergence(cls, p: Dict[str, float], q: Dict[str, float]) -> float:
        """
        Compute Kullback-Leibler divergence between two categorical distributions.
        Measures information gain/disagreement between sources.
        """
        kl_total = 0.0
        for key in ["home", "draw", "away"]:
            kl_total += cls._safe_divide_log(p.get(key, 0), q.get(key, 0))
        return round(kl_total, 4)

    @staticmethod
    def _get_favorite(probs: Dict[str, float]) -> str:
        """Return outcome key with maximum probability."""
        return max(probs, key=probs.get)

    @classmethod
    def compute_divergence_features(
        cls,
        bk_probs: Dict[str, float],
        poly_probs: Dict[str, float],
        ml_probs: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Compute comprehensive divergence features between probability sources.
        High divergence often indicates market inefficiency or insider information.
        """
        features: Dict = {}

        # 1. Raw Probabilities
        for prefix, probs in [("bk", bk_probs), ("poly", poly_probs)]:
            features[f"{prefix}_prob_H"] = probs.get("home", 0.0)
            features[f"{prefix}_prob_D"] = probs.get("draw", 0.0)
            features[f"{prefix}_prob_A"] = probs.get("away", 0.0)

        # 2. KL-Divergence (Bookmaker vs Polymarket)
        features["kl_div_bk_poly"] = cls.compute_kl_divergence(bk_probs, poly_probs)

        # 3. Absolute & Signed Divergences
        max_div = 0.0
        for key, label in [("home", "H"), ("draw", "D"), ("away", "A")]:
            bk = bk_probs.get(key, 0.0)
            poly = poly_probs.get(key, 0.0)
            diff = bk - poly
            abs_diff = abs(diff)
            features[f"divergence_{label}"] = round(diff, 4)
            features[f"abs_divergence_{label}"] = round(abs_diff, 4)
            if abs_diff > max_div:
                max_div = abs_diff

        features["max_divergence"] = round(max_div, 4)

        # 4. Agreement Flags
        bk_fav = cls._get_favorite(bk_probs)
        poly_fav = cls._get_favorite(poly_probs)
        features["bk_fav"] = bk_fav
        features["poly_fav"] = poly_fav
        features["sources_agree"] = int(bk_fav == poly_fav)

        # 5. ML Layer & Triple-Source Blending
        if ml_probs:
            features["ml_prob_H"] = ml_probs.get("home", 0.0)
            features["ml_prob_D"] = ml_probs.get("draw", 0.0)
            features["ml_prob_A"] = ml_probs.get("away", 0.0)

            ml_fav = cls._get_favorite(ml_probs)
            features["ml_fav"] = ml_fav
            features["all_three_agree"] = int(bk_fav == poly_fav == ml_fav)

            # Config-driven weighted blending
            w = TRIPLE_BLEND_WEIGHTS
            for key, label in [("home", "H"), ("draw", "D"), ("away", "A")]:
                ml = ml_probs.get(key, 0.0)
                bk = bk_probs.get(key, 0.0)
                poly = poly_probs.get(key, 0.0)

                features[f"ml_vs_bk_{label}"] = round(ml - bk, 4)
                features[f"ml_vs_poly_{label}"] = round(ml - poly, 4)

                blended = (
                    w["ml"] * ml + 
                    w["polymarket"] * poly + 
                    w["bookmaker"] * bk
                )
                features[f"triple_blend_{label}"] = round(blended, 4)
        else:
            # Fallback to 50/50 blend if ML is unavailable
            for key, label in [("home", "H"), ("draw", "D"), ("away", "A")]:
                bk = bk_probs.get(key, 0.0)
                poly = poly_probs.get(key, 0.0)
                features[f"blended_{label}"] = round(0.5 * bk + 0.5 * poly, 4)
            features["all_three_agree"] = 0

        logger.debug(f"Divergence features computed: {len(features)} keys")
        return features

    @staticmethod
    def compute_liquidity_features(orderbook: Optional[Dict]) -> Dict:
        """
        Extract confidence indicators from Polymarket order book depth.
        Thin order book = noisy signal. Deep order book = strong consensus.
        """
        if not orderbook:
            return {
                "poly_spread": 0.0, "poly_spread_pct": 0.0,
                "poly_depth_total": 0.0, "poly_depth_log": 0.0,
                "poly_imbalance": 0.0, "poly_liquid_market": 0
            }

        spread = orderbook.get("spread", 0.0)
        midpoint = orderbook.get("midpoint", 0.5)
        total_depth = orderbook.get("total_depth", 0.0)
        imbalance = orderbook.get("imbalance", 0.0)

        return {
            "poly_spread": round(spread, 4),
            "poly_spread_pct": round(spread / midpoint, 4) if midpoint > 0 else 0.0,
            "poly_depth_total": round(total_depth, 2),
            "poly_depth_log": round(np.log1p(total_depth), 2),
            "poly_imbalance": round(imbalance, 4),
            "poly_liquid_market": int(total_depth > 5000)  # $5k liquidity threshold
        }

