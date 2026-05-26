"""
Unit tests for SimulatedGANav baseline in scripts/eval_env_baselines.py.

SimulatedGANav replicates the classify-then-act decision logic of GANav
(Guan et al., RA-L 2022) using our label→traversability table.

Key property under test:
  - Group A/B terrain only (τ ≥ 0.50) → PROCEED
  - ANY unknown region present      → STOP (hard refusal)
  - ANY Group C label (τ < 0.50)    → STOP (hard refusal)
  - SimulatedGANav NEVER returns ASK — it never asks a human.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.traversability import TraversabilityMap

H, W = 50, 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def _region(label: str, trav: float, frac: float = 0.50) -> RegionInfo:
    n = int(H * W * frac)
    mask = np.zeros((H, W), dtype=bool)
    mask.flat[:n] = True
    return RegionInfo(
        label=label, mask=mask, confidence=0.85,
        pixel_fraction=frac, source="sam3" if label != "unknown" else "sam2",
        traversability=trav,
    )


def _detection(known_labels, unknown_labels=None, unknown_coverage=0.0):
    """Build a minimal DetectionResult with the given region labels."""
    from system.env_uncertainty.traversability import get_traversability
    known = [_region(lbl, get_traversability(lbl)) for lbl in known_labels]
    unknown = [_region(lbl, 0.0) for lbl in (unknown_labels or [])]
    tmap = TraversabilityMap.create(H, W)
    for r in known:
        tmap = tmap.update_region(r.mask, r.label)
    return DetectionResult(
        known_regions=known,
        unknown_regions=unknown,
        image_shape=(H, W),
        sam3_coverage=sum(r.pixel_fraction for r in known),
        unknown_coverage=unknown_coverage,
        has_unknown=len(unknown) > 0,
        traversability_map=tmap,
    )


def _ganav():
    from scripts.eval_env_baselines import SimulatedGANav
    return SimulatedGANav()


GOAL = (int(H * 0.20), W // 2)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSimulatedGANavProceed:
    """All-safe terrain → PROCEED (Group A/B only)."""

    def test_all_concrete_proceeds(self):
        g = _ganav()
        det = _detection(["concrete"])
        action, _ = g.decide(det, H, W, GOAL)
        assert action == "PROCEED"

    def test_all_grass_proceeds(self):
        g = _ganav()
        det = _detection(["grass"])
        action, _ = g.decide(det, H, W, GOAL)
        assert action == "PROCEED"

    def test_gravel_and_dirt_proceeds(self):
        # Both Group B (τ ≥ 0.50) — should still PROCEED
        g = _ganav()
        det = _detection(["gravel", "dirt"])
        action, _ = g.decide(det, H, W, GOAL)
        assert action == "PROCEED"


class TestSimulatedGANavStop:
    """Non-navigable terrain or unknown regions → STOP, never ASK."""

    def test_unknown_region_stops(self):
        g = _ganav()
        det = _detection(["concrete"], unknown_labels=["unknown"], unknown_coverage=0.20)
        action, _ = g.decide(det, H, W, GOAL)
        assert action == "STOP"

    def test_mud_stops(self):
        # mud has τ=0.10 → Group C
        g = _ganav()
        det = _detection(["mud"])
        action, _ = g.decide(det, H, W, GOAL)
        assert action == "STOP"

    def test_water_stops(self):
        g = _ganav()
        det = _detection(["water"])
        action, _ = g.decide(det, H, W, GOAL)
        assert action == "STOP"

    def test_mixed_safe_and_mud_stops(self):
        # Even one Group C region → STOP regardless of other safe regions
        g = _ganav()
        det = _detection(["concrete", "mud"])
        action, _ = g.decide(det, H, W, GOAL)
        assert action == "STOP"


class TestSimulatedGANavNeverAsks:
    """SimulatedGANav must never return ASK — that is our key differentiator."""

    def test_never_asks_on_safe_terrain(self):
        g = _ganav()
        for label in ["concrete", "grass", "dirt", "gravel"]:
            det = _detection([label])
            action, _ = g.decide(det, H, W, GOAL)
            assert action != "ASK", f"SimulatedGANav asked on '{label}' — should not happen"

    def test_never_asks_on_unknown_terrain(self):
        g = _ganav()
        det = _detection([], unknown_labels=["unknown"], unknown_coverage=0.50)
        action, _ = g.decide(det, H, W, GOAL)
        assert action != "ASK"

    def test_never_asks_on_dangerous_terrain(self):
        g = _ganav()
        det = _detection(["mud", "water"])
        action, _ = g.decide(det, H, W, GOAL)
        assert action != "ASK"
