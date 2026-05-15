"""
FM Traversability Evaluation Harness.

Compares FMTraversabilityScorer accuracy (STATIC vs. FM modes) against
hand-annotated ground-truth traversability scores from RUGD/RELLIS terrain patches.

The primary evaluation metric is Mean Absolute Error (MAE) between the scorer's
predicted value and the human-annotated ground-truth score. Per-class MAE and
two soft-accuracy thresholds (within-0.1, within-0.2) are also reported.

Ground truth JSON format
------------------------
A JSON array where each element has:
  patch_id           (str)   unique identifier, e.g. "rugd_001"
  label              (str)   terrain class string, e.g. "grass"
  scene_context      (str)   free-text scene description (optional)
  ground_truth_score (float) human-annotated traversability in [0, 1]

Usage example
-------------
    from system.env_uncertainty.eval_fm_traversability import (
        load_ground_truth, run_evaluation, compare_modes,
    )
    truth  = load_ground_truth("data/rugd_ground_truth_sample.json")
    result = run_evaluation(truth)             # static mode, no LLM
    print(f"Static MAE: {result.mae:.3f}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from system.env_uncertainty.fm_traversability import (
    FMTraversabilityScorer,
    ScoringMode,
)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class GroundTruthEntry:
    """One annotated terrain patch from RUGD/RELLIS."""
    patch_id: str
    label: str
    scene_context: str
    ground_truth_score: float


@dataclass
class PredictionEntry:
    """Scorer output paired with the ground-truth annotation for one patch."""
    patch_id: str
    label: str
    ground_truth_score: float
    predicted_score: float
    abs_error: float
    source: str
    reasoning: str = ""


@dataclass
class EvaluationResult:
    """
    Aggregate results from evaluating FMTraversabilityScorer against ground truth.

    Attributes:
        mae:           Mean absolute error across all patches.
        per_class_mae: MAE broken down by terrain label.
        predictions:   Per-patch predictions with individual errors.
        mode:          Scoring mode used ("static", "fm", "fm_with_fallback").
        n_patches:     Total number of patches evaluated.
    """

    mae: float
    per_class_mae: Dict[str, float]
    predictions: List[PredictionEntry]
    mode: str
    n_patches: int

    @property
    def rmse(self) -> float:
        """Root-mean-square error over all patches."""
        if not self.predictions:
            return 0.0
        return (
            sum(p.abs_error ** 2 for p in self.predictions) / len(self.predictions)
        ) ** 0.5

    @property
    def within_01(self) -> float:
        """Fraction of patches with absolute error < 0.10."""
        if not self.predictions:
            return 0.0
        return sum(1 for p in self.predictions if p.abs_error < 0.10) / len(
            self.predictions
        )

    @property
    def within_02(self) -> float:
        """Fraction of patches with absolute error < 0.20."""
        if not self.predictions:
            return 0.0
        return sum(1 for p in self.predictions if p.abs_error < 0.20) / len(
            self.predictions
        )

    def summary(self) -> str:
        """One-line human-readable summary of evaluation results."""
        return (
            f"mode={self.mode}  n={self.n_patches}  "
            f"MAE={self.mae:.3f}  RMSE={self.rmse:.3f}  "
            f"within-0.1={self.within_01:.1%}  within-0.2={self.within_02:.1%}"
        )


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_ground_truth(path: str | Path) -> List[GroundTruthEntry]:
    """
    Load human-annotated traversability scores from a JSON file.

    Each JSON element must contain patch_id, label, and ground_truth_score.
    scene_context defaults to "" if omitted.

    Args:
        path: Path to the annotation JSON file.

    Returns:
        Ordered list of GroundTruthEntry instances.

    Raises:
        KeyError: If a required field is missing from any entry.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    raw = json.loads(Path(path).read_text())
    return [
        GroundTruthEntry(
            patch_id=str(item["patch_id"]),
            label=str(item["label"]),
            scene_context=str(item.get("scene_context", "")),
            ground_truth_score=float(item["ground_truth_score"]),
        )
        for item in raw
    ]


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_evaluation(
    entries: List[GroundTruthEntry],
    scorer: Optional[FMTraversabilityScorer] = None,
    mode: ScoringMode = ScoringMode.STATIC,
    llm: Optional[object] = None,
    latency_budget_ms: float = 500.0,
) -> EvaluationResult:
    """
    Score every ground-truth patch and compute MAE and per-class breakdown.

    Args:
        entries:           Annotated patches to evaluate.
        scorer:            Pre-built FMTraversabilityScorer. When provided,
                           mode / llm / latency_budget_ms are ignored.
        mode:              ScoringMode to use when constructing a fresh scorer.
        llm:               LLMInterface instance (required for FM modes).
        latency_budget_ms: Latency budget for FM_WITH_FALLBACK scorer.

    Returns:
        EvaluationResult with MAE, RMSE, per-class errors, and per-patch details.
    """
    if scorer is None:
        scorer = FMTraversabilityScorer(
            llm=llm,
            mode=mode,
            latency_budget_ms=latency_budget_ms,
        )

    predictions: List[PredictionEntry] = []
    class_errors: Dict[str, List[float]] = {}

    for entry in entries:
        judgment = scorer.score_label(entry.label, entry.scene_context)
        err = abs(judgment.score - entry.ground_truth_score)
        predictions.append(
            PredictionEntry(
                patch_id=entry.patch_id,
                label=entry.label,
                ground_truth_score=entry.ground_truth_score,
                predicted_score=judgment.score,
                abs_error=err,
                source=judgment.source,
                reasoning=judgment.reasoning,
            )
        )
        class_errors.setdefault(entry.label, []).append(err)

    mae = (
        sum(p.abs_error for p in predictions) / len(predictions)
        if predictions
        else 0.0
    )
    per_class_mae = {
        label: sum(errs) / len(errs) for label, errs in class_errors.items()
    }

    return EvaluationResult(
        mae=mae,
        per_class_mae=per_class_mae,
        predictions=predictions,
        mode=scorer._mode.value,
        n_patches=len(entries),
    )


def compare_modes(
    entries: List[GroundTruthEntry],
    llm: Optional[object] = None,
    latency_budget_ms: float = 500.0,
) -> Dict[str, EvaluationResult]:
    """
    Evaluate STATIC and FM_WITH_FALLBACK modes side-by-side for comparison.

    If no LLM is provided, only "static" is included in the result dict.
    This supports the paper's Table comparing static-table MAE vs. FM-judge MAE
    over 30 RUGD/RELLIS patches.

    Args:
        entries:           Ground-truth annotations.
        llm:               LLMInterface instance. If None, only static is run.
        latency_budget_ms: Latency budget passed to the FM scorer.

    Returns:
        Dict mapping "static" (and optionally "fm") to EvaluationResult.
    """
    results: Dict[str, EvaluationResult] = {}

    results["static"] = run_evaluation(entries, mode=ScoringMode.STATIC)

    if llm is not None:
        results["fm"] = run_evaluation(
            entries,
            mode=ScoringMode.FM_WITH_FALLBACK,
            llm=llm,
            latency_budget_ms=latency_budget_ms,
        )

    return results
