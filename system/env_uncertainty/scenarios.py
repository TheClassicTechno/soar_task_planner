"""
Concrete test scenarios for the environmental uncertainty pipeline (H33).

Each scenario describes a real-world situation where the robot encounters
terrain uncertainty that it cannot resolve autonomously. These are used for:
  - Qualitative evaluation (does the robot ask the right question?)
  - Unit/integration testing (does the pipeline decide correctly?)
  - Next-meeting demonstration (concrete examples of the pipeline working)

Scenario design principle:
  The robot MUST ask because it cannot determine traversability from its sensor
  readings alone. The user's answer determines whether the robot proceeds.

Five scenarios covering the main uncertainty triggers:
  1. SEMANTIC_UNKNOWN   — SAM2 found a region SAM3 cannot label
  2. SEMANTIC_AMBIGUOUS — SAM3 is uncertain between two similar terrain classes
  3. LOW_TRAVERSABILITY — known terrain class but historically risky (e.g. mud)
  4. SAFE_KNOWN_TERRAIN — control case: robot should PROCEED without asking
  5. FULL_UNKNOWN       — >80% of scene is unknown; robot must STOP

Usage::

    from system.env_uncertainty.scenarios import SCENARIOS, Scenario
    for s in SCENARIOS:
        print(s.name, s.expected_action, s.user_response_example)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Scenario:
    """
    A named test scenario for the environmental uncertainty pipeline.

    name:                  Short identifier.
    description:           Human-readable scene description (1-2 sentences).
    uncertainty_trigger:   Why the robot should ASK or STOP.
    expected_action:       "PROCEED", "ASK", or "STOP".
    unknown_coverage:      Fraction of image that is unknown (0–1).
    terrain_on_path:       Terrain labels the robot's planned path passes through.
    user_response_example: A realistic free-text user answer (for ASK scenarios).
    expected_post_action:  Expected action after user responds (usually PROCEED).
    goal_description:      Where the robot is trying to go.
    """

    name: str
    description: str
    uncertainty_trigger: str
    expected_action: str
    unknown_coverage: float
    terrain_on_path: List[str]
    user_response_example: Optional[str]
    expected_post_action: Optional[str]
    goal_description: str
    source_image: Optional[str] = None


SCENARIOS: List[Scenario] = [

    # ── Scenario 1: SAM2 found a region SAM3 cannot label ────────────────────
    Scenario(
        name="unrecognized_patch",
        description=(
            "The robot is navigating across a park path. A dark region "
            "in the center of the planned path was not identified by SAM3 — "
            "it could be a shadow, a puddle, or a strange surface texture."
        ),
        uncertainty_trigger=(
            "SAM2 detected a region with <30% overlap with SAM3 coverage. "
            "The region is directly on the planned path (passes_through_unknown=True)."
        ),
        expected_action="ASK",
        unknown_coverage=0.25,
        terrain_on_path=["sidewalk", "unknown"],
        user_response_example="It looks like a puddle to me, maybe skip it.",
        expected_post_action="PROCEED",   # robot replans around the puddle
        goal_description="Reach the park bench visible in the top-right of the image.",
        source_image="data/datasets/GOOSE/images/val/2023-05-15_neubiberg_rain/2023-05-15_neubiberg_rain__0691_1684158228061556386_windshield_vis.png",
    ),

    # ── Scenario 2: Semantic ambiguity — mud vs gravel ────────────────────────
    Scenario(
        name="mud_or_gravel_ambiguity",
        description=(
            "The robot is on a trail. Ahead is a brownish surface that SAM3 "
            "classified as either 'mud' (traversability=0.10) or 'gravel' "
            "(traversability=0.70). The Dirichlet entropy for this node is 1.8, "
            "above the ask threshold of 1.5."
        ),
        uncertainty_trigger=(
            "Dirichlet semantic entropy > entropy_ask_threshold (1.8 > 1.5). "
            "Top-3 candidates: mud (42%), gravel (38%), dirt (20%). "
            "The robot cannot distinguish because the color and texture overlap."
        ),
        expected_action="ASK",
        unknown_coverage=0.05,
        terrain_on_path=["mud", "gravel"],
        user_response_example="That's gravel, it's dry and solid, safe to cross.",
        expected_post_action="PROCEED",
        goal_description="Reach the trailhead marker 30 metres ahead.",
        source_image="data/datasets/GOOSE/images/val/2022-07-22_flight/2022-07-22_flight__0194_1658494592194049657_windshield_vis.png",
    ),

    # ── Scenario 3: Known terrain class but low GP LCB ───────────────────────
    Scenario(
        name="wet_grass_low_lcb",
        description=(
            "A grass lawn after rainfall. SAM3 correctly identifies it as 'grass'. "
            "The real GT image has ~3.5% unlabeled pixels (below path_unknown_tolerance), "
            "so coverage is noise-level and the pipeline proceeds without asking. "
            "NOTE: GP LCB-based STOP for known-but-wet terrain is aspirational — it "
            "requires dense multi-point GP seeding and dynamic traversability scoring "
            "based on visual wetness cues, neither of which is implemented yet."
        ),
        uncertainty_trigger=(
            "unknown_coverage=0.035 < path_unknown_tolerance=0.06: treated as GT "
            "labeling noise. LCB STOP is skipped at noise-level coverage because "
            "sparse single-centroid GP seeding gives unreliable estimates. "
            "Robot proceeds since no ASK/STOP condition triggers."
        ),
        expected_action="PROCEED",
        unknown_coverage=0.035,
        terrain_on_path=["grass"],
        user_response_example=None,
        expected_post_action=None,
        goal_description="Cross the lawn to reach the building entrance.",
        source_image="data/datasets/GOOSE/images/val/2023-03-03_garching_2/2023-03-03_garching_2__0207_1677850967765427698_windshield_vis.png",
    ),

    # ── Scenario 4: Control — safe known terrain, no asking needed ───────────
    Scenario(
        name="clear_sidewalk",
        description=(
            "The robot is on a paved sidewalk. SAM3 identifies all visible terrain "
            "as 'sidewalk' or 'concrete'. No unknown regions. GP LCB is 0.82. "
            "Dirichlet entropy for all on-path nodes is 0.3 (very confident)."
        ),
        uncertainty_trigger=(
            "Minimal unknown coverage (~1.2% GT labeling noise). GP LCB well above "
            "threshold. Entropy is low. Coverage is below path_unknown_tolerance — "
            "robot treats unlabeled pixels as noise and proceeds."
        ),
        expected_action="PROCEED",
        unknown_coverage=0.012,
        terrain_on_path=["sidewalk", "concrete"],
        user_response_example=None,       # no question needed
        expected_post_action=None,
        goal_description="Continue along the sidewalk toward the intersection.",
        source_image="data/datasets/GOOSE/images/val/2022-12-07_aying_hills/2022-12-07_aying_hills__0029_1670421161282631132_windshield_vis.png",
    ),

    # ── Scenario 5: Dry gravel path — no unknowns, safe to proceed ──────────
    Scenario(
        name="dry_gravel_path",
        description=(
            "The robot is on an unpaved trail with visible gravel and compacted dirt. "
            "SAM3 labels the entire forward view as 'gravel' or 'dirt'. No unknown "
            "regions. GP LCB = 0.65 from prior dry-weather observations."
        ),
        uncertainty_trigger=(
            "Minimal unknown coverage (~3.7% GT labeling noise). GP LCB above "
            "lcb_stop_threshold=0.05 for gravel/dirt terrain. Coverage is below "
            "path_unknown_tolerance — robot treats unlabeled pixels as noise and proceeds."
        ),
        expected_action="PROCEED",
        unknown_coverage=0.037,
        terrain_on_path=["gravel", "dirt"],
        user_response_example=None,
        expected_post_action=None,
        goal_description="Reach the park entrance 40 metres ahead along the gravel path.",
        source_image="data/datasets/GOOSE/images/val/2023-03-03_garching_2/2023-03-03_garching_2__0208_1677850973351100270_windshield_vis.png",
    ),

    # ── Scenario 6: Campus crosswalk — well-observed, high GP confidence ─────
    Scenario(
        name="campus_crosswalk",
        description=(
            "The robot is crossing a paved university campus path. SAM3 identifies "
            "the entire forward view as 'concrete' and 'crosswalk' markings. The GP "
            "has 5 prior observations at this location, all confirming high traversability. "
            "GP LCB = 0.88. Dirichlet entropy = 0.2."
        ),
        uncertainty_trigger=(
            "Minimal unknown coverage (~5.5% GT labeling noise). GP LCB above "
            "lcb_stop_threshold for concrete terrain. Coverage is below "
            "path_unknown_tolerance — robot treats unlabeled pixels as noise and proceeds."
        ),
        expected_action="PROCEED",
        unknown_coverage=0.055,
        terrain_on_path=["concrete", "crosswalk"],
        user_response_example=None,
        expected_post_action=None,
        goal_description="Cross to the building entrance on the other side of the path.",
        source_image="data/datasets/GOOSE/images/val/2022-09-21_garching_uebungsplatz_2/2022-09-21_garching_uebungsplatz_2__0122_1663756023767112887_windshield_vis.png",
    ),

    # ── Scenario 7: Scene overwhelmingly unknown — STOP ──────────────────────
    Scenario(
        name="flooded_trail",
        description=(
            "The robot encounters a section of trail that is heavily flooded. "
            "SAM2 finds many regions; SAM3 matches less than 20% of them. "
            "Unknown coverage is 0.85. No candidate trajectory avoids the unknown area."
        ),
        uncertainty_trigger=(
            "unknown_coverage=0.387 >= large_unknown_stop_threshold=0.30. "
            "Coverage is large enough that even a technically safe-scored path "
            "cannot be trusted — robot stops before asking."
        ),
        expected_action="STOP",
        unknown_coverage=0.387,
        terrain_on_path=["unknown", "water"],
        user_response_example=(
            "The trail is flooded, do not cross. Turn around and take the paved path."
        ),
        expected_post_action="STOP",    # even after user confirms, no safe path
        goal_description="Reach the picnic area at the end of the trail.",
        source_image="data/datasets/GOOSE/images/val/2023-01-20_aying_mangfall_2/2023-01-20_aying_mangfall_2__0434_1674223443460061906_windshield_vis.png",
    ),
]


def get_scenario(name: str) -> Optional[Scenario]:
    """Return the scenario with the given name, or None if not found."""
    return next((s for s in SCENARIOS if s.name == name), None)


def ask_scenarios() -> List[Scenario]:
    """Return only scenarios where the robot should ASK."""
    return [s for s in SCENARIOS if s.expected_action == "ASK"]


def stop_scenarios() -> List[Scenario]:
    """Return only scenarios where the robot should STOP."""
    return [s for s in SCENARIOS if s.expected_action == "STOP"]


def proceed_scenarios() -> List[Scenario]:
    """Return only scenarios where the robot should PROCEED."""
    return [s for s in SCENARIOS if s.expected_action == "PROCEED"]
