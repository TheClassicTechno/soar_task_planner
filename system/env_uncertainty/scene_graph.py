"""
Scene graph with terrain memory for outdoor robot navigation.

Maintains a coarse 10×10 grid of TerrainNodes, one per (terrain_label, cell)
pair.  The graph serves as a memory: once a terrain region is confirmed by the
user AND the GP variance is low, should_skip_asking() returns True so the robot
does not re-ask about the same terrain in the same location.

Design
------
  • Grid: 10×10 cells over the image. Each cell (cy, cx) ∈ [0,9]².
  • Key: (label.lower(), (cy, cx)) → TerrainNode.
  • Node states: UNKNOWN → INFERRED (after GP update) → CONFIRMED (after user).
  • Skip condition: user_confirmed=True AND gp_variance < SKIP_VARIANCE_THRESHOLD.

Semantic uncertainty (Dirichlet distribution)
---------------------------------------------
  Each node carries a Dirichlet concentration vector alpha[K] over the full
  terrain vocabulary.  This models *which class the terrain is*, separately from
  the GP which models *how traversable it is*.

  Prior: uniform Dir(1,...,1) = equal uncertainty about all classes.
  Update: alpha[matched_class] += label_confidence  (conjugate prior, O(K)).
  Posterior mean: E[X_i] = alpha_i / sum(alpha).
  Entropy: high = uncertain about terrain type; triggers ASK in runner.

  TerrainNodeV2 is a module-level alias for TerrainNode for forward compatibility.

Integration
-----------
  update_from_gp() pulls GP posterior means + variances into existing nodes.
  mark_confirmed() records the user's safe/unsafe decision.
  TerrainNode.update_from_user() performs the Dirichlet conjugate update.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from system.env_uncertainty.gp_traversability import GPTraversabilityMap

import numpy as np

from system.env_uncertainty.traversability import (
    get_traversability,
    STOP_THRESHOLD,
    TRAVERSABILITY_SCORES,
)

# Ordered list of terrain classes matching TRAVERSABILITY_SCORES.
# Order is stable so dirichlet_alpha indices are always consistent.
TERRAIN_CLASSES: List[str] = list(TRAVERSABILITY_SCORES.keys())
K_CLASSES: int = len(TERRAIN_CLASSES)


def _uniform_alpha() -> List[float]:
    """Return a flat Dirichlet prior: equal uncertainty over all terrain classes."""
    return [1.0] * K_CLASSES


class CertaintyLevel(Enum):
    UNKNOWN = "unknown"
    INFERRED = "inferred"
    CONFIRMED = "confirmed"


@dataclass
class TerrainNode:
    """
    One terrain region in the scene graph.

    label:                  Terrain class (e.g. "grass", "unknown").
    position_cell_id:       (cy, cx) coarse grid cell in [0, GRID_SIZE-1]².
    gp_mean:                GP posterior mean traversability ∈ [0, 1].
    gp_variance:            GP posterior variance ∈ [0, ∞).
    certainty_level:        UNKNOWN / INFERRED / CONFIRMED.
    user_confirmed:         True after user has explicitly responded.
    dirichlet_alpha:        Dirichlet concentration vector over TERRAIN_CLASSES.
                            Encodes semantic uncertainty: which class is this?
                            Updated via conjugate prior when user responds.
    adjacent_trajectory_ids: Trajectory names that pass through this cell.
    last_updated:           Unix timestamp of last update.
    """

    label: str
    position_cell_id: Tuple[int, int]
    gp_mean: float
    gp_variance: float
    certainty_level: CertaintyLevel
    user_confirmed: bool
    dirichlet_alpha: List[float] = field(default_factory=_uniform_alpha)
    adjacent_trajectory_ids: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    # ── Dirichlet semantic distribution methods ───────────────────────────────

    def semantic_distribution(self) -> np.ndarray:
        """
        Expected class probabilities: E[X_i] = alpha_i / sum(alpha).

        Returns a length-K probability vector summing to 1.
        """
        a = np.array(self.dirichlet_alpha, dtype=float)
        return a / a.sum()

    def map_class(self) -> str:
        """Most probable terrain class according to the Dirichlet posterior."""
        return TERRAIN_CLASSES[int(np.argmax(self.semantic_distribution()))]

    def semantic_entropy(self) -> float:
        """
        Shannon entropy of the expected class distribution: H = -Σ p_i log p_i.

        p_i = alpha_i / sum(alpha)  (the Dirichlet posterior mean).

        Range: [0, log(K)].  0 = certain about one class; log(K) = uniform.
        Used as the ASK trigger: high entropy means the robot does not know what
        terrain type it is looking at, regardless of the GP traversability estimate.

        Note: this is NOT the differential entropy of Dir(alpha).  The differential
        entropy grows as pseudocounts accumulate (more data = larger log-normaliser),
        which is counter-intuitive for a trigger threshold.  The Shannon entropy of
        the mean is the right measure: it falls monotonically toward 0 as alpha
        concentrates on one class.
        """
        p = self.semantic_distribution()
        p_safe = np.where(p > 0, p, 1.0)  # avoid log(0); zero-prob terms drop out
        return float(-np.sum(p * np.log(p_safe)))

    def expected_traversability(self) -> float:
        """
        Probability-weighted expected traversability from Dirichlet posterior.

        More principled than a hard label lookup: accounts for uncertainty
        about which terrain class this region actually is.

        E[trav] = sum_i P(class=i) * TRAVERSABILITY_SCORES[class_i]
        """
        p = self.semantic_distribution()
        scores = np.array(
            [TRAVERSABILITY_SCORES.get(cls, 0.0) for cls in TERRAIN_CLASSES],
            dtype=float,
        )
        return float(np.dot(p, scores))

    def update_from_user(self, terrain_label: str, confidence: float) -> None:
        """
        Dirichlet conjugate prior update from user terrain feedback.

        Posterior: alpha_new[i] += confidence for the matched class i.
        This is the closed-form Bayesian update — O(1), no MCMC.

        Args:
            terrain_label: User-provided terrain class (must be in TERRAIN_CLASSES).
            confidence:    User's stated confidence in [0, 1].
        """
        label_lower = terrain_label.lower()
        if label_lower not in TERRAIN_CLASSES:
            return
        idx = TERRAIN_CLASSES.index(label_lower)
        self.dirichlet_alpha[idx] += float(confidence)
        self.last_updated = time.time()

    def top_k_classes(self, k: int = 3) -> List[Tuple[str, float]]:
        """
        Top-k terrain classes by posterior probability (for VLM question context).

        Returns:
            List of (class_name, probability) sorted descending by probability.
        """
        p = self.semantic_distribution()
        top_idx = np.argsort(p)[::-1][:k]
        return [(TERRAIN_CLASSES[i], float(p[i])) for i in top_idx]


# Forward-compatible alias — new code should use TerrainNodeV2.
TerrainNodeV2 = TerrainNode


class SceneGraph:
    """
    Coarse terrain memory over the image plane.

    Nodes are keyed by (label.lower(), (cy, cx)) so different terrain types
    in the same cell are tracked separately.

    Class constants
    ---------------
    GRID_SIZE:               Number of cells per axis (10 → 10×10 grid).
    SKIP_VARIANCE_THRESHOLD: σ² below this + user_confirmed → skip asking.
    """

    GRID_SIZE: int = 10
    SKIP_VARIANCE_THRESHOLD: float = 0.04  # σ < 0.2

    def __init__(self) -> None:
        self._nodes: Dict[Tuple[str, Tuple[int, int]], TerrainNode] = {}

    # ── Coordinate mapping ────────────────────────────────────────────────────

    def pixel_to_cell(
        self,
        pixel_y: int,
        pixel_x: int,
        height: int,
        width: int,
    ) -> Tuple[int, int]:
        """
        Map a pixel coordinate to the coarse (cy, cx) grid cell.

        Args:
            pixel_y: Row index (0-indexed).
            pixel_x: Column index (0-indexed).
            height:  Image height in pixels.
            width:   Image width in pixels.

        Returns:
            (cy, cx) with cy, cx ∈ [0, GRID_SIZE-1].
        """
        cy = min(int(pixel_y * self.GRID_SIZE / max(height, 1)), self.GRID_SIZE - 1)
        cx = min(int(pixel_x * self.GRID_SIZE / max(width, 1)), self.GRID_SIZE - 1)
        return cy, cx

    # ── Node management ───────────────────────────────────────────────────────

    def upsert_region(
        self,
        label: str,
        pixel_y: int,
        pixel_x: int,
        height: int,
        width: int,
        gp_mean: Optional[float] = None,
        gp_variance: Optional[float] = None,
    ) -> TerrainNode:
        """
        Create or update the TerrainNode for (label, cell).

        If the node already exists, updates gp_mean/gp_variance (if provided)
        and last_updated.  Does not downgrade certainty_level or user_confirmed.

        Args:
            label:       Terrain class label.
            pixel_y:     Representative row pixel for the region.
            pixel_x:     Representative column pixel for the region.
            height:      Image height.
            width:       Image width.
            gp_mean:     Override GP mean (None = use get_traversability default).
            gp_variance: Override GP variance (None = use 0.0 default).

        Returns:
            The created or updated TerrainNode.
        """
        key = self._key(label, pixel_y, pixel_x, height, width)
        cell = self.pixel_to_cell(pixel_y, pixel_x, height, width)

        if key in self._nodes:
            node = self._nodes[key]
            if gp_mean is not None:
                node.gp_mean = gp_mean
            if gp_variance is not None:
                node.gp_variance = gp_variance
            node.last_updated = time.time()
            if node.certainty_level == CertaintyLevel.UNKNOWN and (
                gp_mean is not None or gp_variance is not None
            ):
                node.certainty_level = CertaintyLevel.INFERRED
            return node

        initial_mean = gp_mean if gp_mean is not None else get_traversability(label)
        initial_var = gp_variance if gp_variance is not None else 0.0
        level = CertaintyLevel.INFERRED if gp_mean is not None else CertaintyLevel.UNKNOWN

        # Label-informed Dirichlet prior: SAM3's detected class gets pseudocount 2.0;
        # all others start at 1.0.  This encodes mild confidence in SAM3's label
        # while still allowing the user to override via update_from_user().
        initial_alpha = _uniform_alpha()
        label_lower = label.lower()
        if label_lower in TERRAIN_CLASSES:
            initial_alpha[TERRAIN_CLASSES.index(label_lower)] = 2.0

        node = TerrainNode(
            label=label_lower,
            position_cell_id=cell,
            gp_mean=initial_mean,
            gp_variance=initial_var,
            certainty_level=level,
            user_confirmed=False,
            dirichlet_alpha=initial_alpha,
        )
        self._nodes[key] = node
        return node

    def recall(
        self,
        label: str,
        cy: int,
        cx: int,
    ) -> Optional[TerrainNode]:
        """
        Look up a TerrainNode by label and cell coordinates.

        Args:
            label: Terrain class label.
            cy:    Grid row ∈ [0, GRID_SIZE-1].
            cx:    Grid column ∈ [0, GRID_SIZE-1].

        Returns:
            TerrainNode if found, else None.
        """
        return self._nodes.get((label.lower(), (cy, cx)))

    def should_skip_asking(
        self,
        label: str,
        cy: int,
        cx: int,
    ) -> Tuple[bool, Optional[TerrainNode]]:
        """
        Return whether we can skip asking the user about this terrain region.

        Skip condition: user_confirmed=True AND gp_variance < SKIP_VARIANCE_THRESHOLD.

        Args:
            label: Terrain class label.
            cy:    Grid row.
            cx:    Grid column.

        Returns:
            (should_skip, node_or_None)
        """
        node = self.recall(label, cy, cx)
        if node is None:
            return False, None
        skip = node.user_confirmed and node.gp_variance < self.SKIP_VARIANCE_THRESHOLD
        return skip, node

    def mark_confirmed(
        self,
        label: str,
        cy: int,
        cx: int,
        is_traversable: bool,
    ) -> Optional[TerrainNode]:
        """
        Record that the user has confirmed traversability for this region.

        Sets user_confirmed=True and certainty_level=CONFIRMED.

        Args:
            label:          Terrain class label.
            cy:             Grid row.
            cx:             Grid column.
            is_traversable: User's answer (not stored in node — GP handles the score).

        Returns:
            Updated TerrainNode, or None if not found.
        """
        node = self.recall(label, cy, cx)
        if node is None:
            return None
        node.user_confirmed = True
        node.certainty_level = CertaintyLevel.CONFIRMED
        node.last_updated = time.time()
        return node

    def update_from_gp(
        self,
        gp_map: "GPTraversabilityMap",
        regions: list,
        height: int,
        width: int,
        trajectories: Optional[list] = None,
    ) -> None:
        """
        Pull GP posterior mean + variance into all region nodes.

        For each RegionInfo in regions, finds the representative pixel
        (centroid of the mask), queries the GP, and upserts the node.

        Args:
            gp_map:      GPTraversabilityMap instance.
            regions:     List of RegionInfo objects.
            height:      Image height.
            width:       Image width.
            trajectories: Optional list of Trajectory objects; their names are
                          recorded on nodes whose cells the trajectory passes through.
        """
        for region in regions:
            cy_rep, cx_rep = self._mask_centroid(region.mask, height, width)
            cell = self.pixel_to_cell(cy_rep, cx_rep, height, width)
            pred = gp_map.predict(cy_rep, cx_rep, height, width)
            node = self.upsert_region(
                label=region.label,
                pixel_y=cy_rep,
                pixel_x=cx_rep,
                height=height,
                width=width,
                gp_mean=pred.mu,
                gp_variance=pred.sigma ** 2,
            )

            if trajectories:
                for traj in trajectories:
                    for wy, wx in traj.waypoints:
                        traj_cell = self.pixel_to_cell(wy, wx, height, width)
                        if traj_cell == cell and traj.name not in node.adjacent_trajectory_ids:
                            node.adjacent_trajectory_ids.append(traj.name)

    def nodes_in_cell(self, cy: int, cx: int) -> List[TerrainNode]:
        """
        Return all TerrainNodes whose position_cell_id is (cy, cx).

        Used by the runner to find every terrain label at a trajectory waypoint
        without knowing the label in advance.

        Args:
            cy: Grid row ∈ [0, GRID_SIZE-1].
            cx: Grid column ∈ [0, GRID_SIZE-1].

        Returns:
            List of TerrainNodes (may be empty).
        """
        return [
            node for (_, cell), node in self._nodes.items() if cell == (cy, cx)
        ]

    @property
    def node_count(self) -> int:
        """Total number of TerrainNodes in the graph."""
        return len(self._nodes)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _key(
        self,
        label: str,
        pixel_y: int,
        pixel_x: int,
        height: int,
        width: int,
    ) -> Tuple[str, Tuple[int, int]]:
        cell = self.pixel_to_cell(pixel_y, pixel_x, height, width)
        return (label.lower(), cell)

    @staticmethod
    def _mask_centroid(mask: object, height: int, width: int) -> Tuple[int, int]:
        """Return the (y, x) centroid pixel of a boolean mask."""
        import numpy as np
        arr = np.asarray(mask)
        if not np.any(arr):
            return height // 2, width // 2
        ys, xs = np.where(arr)
        return int(np.mean(ys)), int(np.mean(xs))
