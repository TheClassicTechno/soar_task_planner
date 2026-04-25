"""
Conformal prediction wrapper for the IntroPlan baseline.

Adapted from Liang et al. NeurIPS 2024 "Introspective Planning" §3.3.

How it works:
  1. CALIBRATION (offline): Given N calibration scenarios with known correct
     options, collect Claude's stated confidence for the correct option on
     each scenario. Compute the (1 - alpha) quantile of (1 - confidence) scores.
     This gives the threshold tau.

  2. INFERENCE (online): Given a new scenario, Claude predicts each option's
     confidence. Include all options whose confidence >= (1 - tau) in the
     prediction set.
     - |prediction set| == 1 → robot ACTS (that option is predicted)
     - |prediction set| > 1  → robot ASKS (too uncertain to commit)

The conformal guarantee: with probability >= (1 - alpha), the correct option
is included in the prediction set. This is the same statistical guarantee as
the original IntroPlan paper's conformal prediction variant.

Note on confidence: since Claude does not expose token log probabilities, we
use Claude's stated confidence score (0.0–1.0 float in the JSON response).
This is an approximation; the original paper uses token-level log probs.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class ConformalPredictor:
    """
    Calibrates and applies conformal prediction for navigation option selection.

    Usage (calibration):
        predictor = ConformalPredictor(alpha=0.15)
        for scenario in calibration_set:
            predictor.record_calibration(scenario["confidence"], scenario["correct_option"])
        predictor.calibrate()

    Usage (inference):
        pred_set = predictor.predict_set({"A": 0.8, "B": 0.3, "C": 0.1, "D": 0.2})
        if len(pred_set) == 1:
            action = pred_set[0]  # act
        else:
            action = "ASK"        # ask user for clarification
    """

    def __init__(self, alpha: float = 0.15):
        """
        Args:
            alpha: Miscoverage level. Target success rate = 1 - alpha.
                   alpha=0.15 means we guarantee 85% coverage (same as IntroPlan paper).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self._nonconformity_scores: List[float] = []  # 1 - confidence_correct_option
        self._tau: Optional[float] = None  # threshold set after calibrate()

    def record_calibration(
        self,
        option_confidences: Dict[str, float],
        correct_option: str,
    ) -> None:
        """
        Record one calibration datapoint.

        Args:
            option_confidences: {"A": 0.8, "B": 0.2, "C": 0.1, "D": 0.05}
            correct_option:     Ground-truth correct option label (e.g., "B").
        """
        confidence_correct = option_confidences.get(correct_option, 0.0)
        # Nonconformity score = 1 - confidence for the correct option.
        # High score = model was uncertain about the correct answer (bad).
        self._nonconformity_scores.append(1.0 - confidence_correct)

    def calibrate(self) -> float:
        """
        Compute the conformal threshold tau from recorded calibration scores.

        tau = ceil((n+1)(1-alpha)) / n  quantile of the nonconformity scores.
        This is the standard finite-sample conformal quantile from Angelopoulos & Bates 2021.

        Returns:
            The computed threshold tau.

        Raises:
            RuntimeError: If no calibration data has been recorded.
        """
        if not self._nonconformity_scores:
            raise RuntimeError(
                "No calibration data recorded. Call record_calibration() first."
            )

        n = len(self._nonconformity_scores)
        scores = np.array(self._nonconformity_scores)

        # Finite-sample adjusted quantile level
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = float(np.clip(level, 0.0, 1.0))
        self._tau = float(np.quantile(scores, level))
        return self._tau

    def predict_set(self, option_confidences: Dict[str, float]) -> List[str]:
        """
        Given option confidences for a new scenario, return the prediction set.

        An option is included in the set if its confidence >= (1 - tau).
        The set is sorted by decreasing confidence.

        Args:
            option_confidences: {"A": 0.8, "B": 0.2, "C": 0.1, "D": 0.05}

        Returns:
            List of option labels in the prediction set (e.g., ["A"] or ["A", "B"]).
            Empty if no option exceeds threshold (treated as "ASK").

        Raises:
            RuntimeError: If calibrate() has not been called yet.
        """
        if self._tau is None:
            raise RuntimeError("Call calibrate() before predict_set().")

        # Small epsilon guards against floating-point issues at boundary
        # (e.g., 1.0 - 0.7 = 0.30000000000000004 > 0.3 in IEEE 754).
        threshold = 1.0 - self._tau - 1e-9
        included = [
            opt for opt, conf in option_confidences.items()
            if conf >= threshold
        ]
        # Sort by confidence descending so the most likely option is first
        included.sort(key=lambda o: option_confidences[o], reverse=True)
        return included

    def should_ask(self, option_confidences: Dict[str, float]) -> bool:
        """
        Returns True if the robot should ask the user (prediction set has >1 option),
        False if it should act (prediction set has exactly 1 option).
        """
        pred_set = self.predict_set(option_confidences)
        return len(pred_set) != 1

    @property
    def tau(self) -> Optional[float]:
        """The calibrated threshold. None before calibrate() is called."""
        return self._tau

    @property
    def n_calibration(self) -> int:
        """Number of calibration samples recorded."""
        return len(self._nonconformity_scores)

    def save(self, path: str) -> None:
        """Save calibration state (scores + tau) to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "alpha": self.alpha,
                "tau": self._tau,
                "nonconformity_scores": self._nonconformity_scores,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConformalPredictor":
        """Load calibration state from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        predictor = cls(alpha=data["alpha"])
        predictor._nonconformity_scores = data["nonconformity_scores"]
        predictor._tau = data.get("tau")
        return predictor


# ── Utility: normalize raw confidences from LLM ──────────────────────────────

def normalize_confidences(raw: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize option confidences to sum to 1.0.

    Claude's stated confidence values are not guaranteed to be calibrated
    or to sum to 1. Normalizing makes threshold comparison consistent.

    Args:
        raw: {"A": 0.9, "B": 0.3, "C": 0.1, "D": 0.2}

    Returns:
        Same keys, values normalized to sum to 1.0.
    """
    total = sum(raw.values())
    if total <= 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw}
    return {k: v / total for k, v in raw.items()}


def extract_option_confidences(llm_prediction: Dict) -> Dict[str, float]:
    """
    Extract per-option confidence scores from an IntroPlan LLM response.

    The LLM response has:
      - "prediction": "B"          (best option)
      - "confidence": 0.85         (confidence in the prediction)
      - "reasoning": {"A": ..., "B": ..., "C": ..., "D": ...}

    Since Claude gives one overall confidence, we:
      - Assign the stated confidence to the predicted option
      - Distribute the remaining (1 - confidence) equally among others

    This approximates the per-option probability distribution for conformal prediction.

    Args:
        llm_prediction: Parsed JSON from format_introspective_predict_prompt.

    Returns:
        {"A": float, "B": float, "C": float, "D": float}
    """
    prediction = llm_prediction.get("prediction", "A").strip().upper()
    confidence = float(llm_prediction.get("confidence", 0.5))
    confidence = float(np.clip(confidence, 0.01, 0.99))

    options = ["A", "B", "C", "D"]
    others = [o for o in options if o != prediction]
    remaining = (1.0 - confidence) / len(others) if others else 0.0

    return {
        o: (confidence if o == prediction else remaining)
        for o in options
    }
