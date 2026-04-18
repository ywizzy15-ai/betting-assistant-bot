"""
grading.py - Confidence scoring and divergence analysis module.
Implements the triple-layer probability blending and value grading logic.
"""
import math
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class ConfidenceGrader:
    def __init__(self):
        # Grading thresholds
        self.grade_levels = {
            "HIGH": 15.0,
            "MEDIUM": 5.0
        }

    def calculate_confidence(
        self,
        ml_prob: float,
        bk_prob: float,
        poly_prob: Optional[float] = None,
        poly_liquidity: Optional[float] = None
    ) -> Dict:
        """
        Calculate confidence score based on triple-layer divergence.
        
        Args:
            ml_prob: Probability from ML model (0.0 to 1.0)
            bk_prob: Probability from Bookmaker odds (0.0 to 1.0)
            poly_prob: Probability from Polymarket (0.0 to 1.0)
            poly_liquidity: Total liquidity depth in USD
            
        Returns:
            Dict with confidence_score, grade, and divergence metrics
        """
        # 1. Base Edge: How much the ML model disagrees with Bookmaker (Value)
        ml_bk_gap = abs(ml_prob - bk_prob)
        edge_score = ml_bk_gap * 100  # Scale to 0-100 range

        # 2. Polymarket Divergence & Agreement
        poly_agreement_bonus = 0.0
        divergence_status = "🟢 Agreement"
        
        if poly_prob is not None:
            ml_poly_gap = abs(ml_prob - poly_prob)
            bk_poly_gap = abs(bk_prob - poly_prob)
            
            # If Polymarket agrees with ML but disagrees with Bookmaker -> High Value Signal
            if ml_poly_gap < 0.05 and bk_poly_gap > 0.08:
                poly_agreement_bonus = 5.0
                divergence_status = "🟡 Moderate / ML-Poly Agreement"
            # If Polymarket disagrees with both -> Potential risk or insider info
            elif ml_poly_gap > 0.10:
                poly_agreement_bonus = -2.0
                divergence_status = "🔴 High Divergence"
            # If all three agree -> Safe bet, lower edge but high probability
            else:
                divergence_status = "🟢 Agreement"
        else:
            divergence_status = "No Poly Data"

        # 3. Liquidity Factor (Polymarket)
        # Logarithmic scaling to prevent outliers
        liquidity_score = 0.0
        if poly_liquidity and poly_liquidity > 0:
            liquidity_score = math.log1p(poly_liquidity) / 10.0 
            # Threshold check: > $5k is significant
            if poly_liquidity > 5000:
                liquidity_score += 2.0
        liquidity_score = min(liquidity_score, 5.0) # Cap at 5 points

        # 4. Composite Score Calculation
        confidence_score = (
            (edge_score * 1.0) + 
            poly_agreement_bonus + 
            liquidity_score - 
            3.0 # Base adjustment to normalize distribution
        )
        
        # Clamp score
        confidence_score = max(0.0, min(100.0, confidence_score))

        # 5. Determine Grade
        if confidence_score >= self.grade_levels["HIGH"]:
            grade = "HIGH"
            label = "🟢 High Value"
        elif confidence_score >= self.grade_levels["MEDIUM"]:
            grade = "MEDIUM"
            label = "🟡 Medium Value"
        else:
            grade = "LOW"
            label = "🔴 Low Edge"

        # 6. KL-Divergence (Advanced metric)
        kl_div = self._compute_kl_divergence(bk_prob, poly_prob or bk_prob)

        result = {
            "confidence_score": round(confidence_score, 2),
            "grade": grade,
            "label": label,
            "divergence_status": divergence_status,
            "edge_percentage": round((ml_bk_gap * 100), 2),
            "kl_divergence": round(kl_div, 4),
            "liquidity_score": round(liquidity_score, 2)
        }

        logger.debug(f"Grading Result: {result}")
        return result

    def _compute_kl_divergence(self, p: float, q: float) -> float:
        """
        Compute Kullback-Leibler divergence between two probabilities.
        Used as a measure of information gain/disagreement.
        """
        epsilon = 1e-9
        p = max(p, epsilon)
        q = max(q, epsilon)
        return p * math.log(p / q) + (1-p) * math.log((1-p) / (1-q))

# Singleton instance
grader = ConfidenceGrader()

