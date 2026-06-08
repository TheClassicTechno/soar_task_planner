"""
Combined environmental + instructional uncertainty scenarios.

Per june1actionitems.txt item 20: scenarios where both uncertainty types
interact, covering three combinations:
  Type A — Ambiguous instruction + uncertain terrain (both branches fire)
  Type B — Clear instruction + uncertain terrain (env branch dominates)
  Type C — Ambiguous instruction + clear terrain (instruction branch dominates)

Each scenario defines:
  instruction:        the user's command to the robot
  terrain_description: what the robot sees
  unknown_coverage:   fraction of image in unknown regions
  expected_dominant:  which branch should dominate ("instruction"/"environment"/"none")
  expected_action:    "ASK", "PROCEED", or "STOP"
  expected_question_about: "instruction" or "terrain" (which kind of Q the robot asks)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CombinedScenario:
    name: str
    instruction: str
    terrain_description: str
    unknown_coverage: float
    expected_dominant: str
    expected_action: str
    expected_question_about: str
    description: str


COMBINED_SCENARIOS = [
    # ── Type A: Ambiguous instruction + uncertain terrain ─────────────────────
    CombinedScenario(
        name="ambiguous_target_unknown_terrain",
        instruction="Go to that thing",
        terrain_description="Dark irregular patch ahead of unknown material, "
                            "standard paved path also available.",
        unknown_coverage=0.25,
        expected_dominant="instruction",   # κ_I=0.75 > κ_E=0.31
        expected_action="ASK",
        expected_question_about="instruction",
        description="User points vaguely ('that thing') AND terrain ahead is "
                    "25% unidentified. Instruction ambiguity (κ_I=0.75) should "
                    "dominate over env uncertainty (κ_E=0.31). Robot asks about "
                    "the instruction, not the terrain.",
    ),

    # ── Type B: Clear instruction + uncertain terrain ─────────────────────────
    CombinedScenario(
        name="clear_instruction_uncertain_path",
        instruction="Go straight ahead to the bench",
        terrain_description="Bench visible at 4m. Unknown wet surface covering "
                            "30% of the direct path.",
        unknown_coverage=0.30,
        expected_dominant="environment",   # κ_I≈0 < κ_E=0.38
        expected_action="ASK",
        expected_question_about="terrain",
        description="Instruction is unambiguous (clear target: bench). Terrain "
                    "ahead is 30% unknown. Env branch dominates (κ_E=0.38 > κ_I≈0). "
                    "Robot asks about the terrain, not the instruction.",
    ),

    # ── Type C: Ambiguous instruction + clear terrain ─────────────────────────
    CombinedScenario(
        name="ambiguous_target_clear_path",
        instruction="Go there",
        terrain_description="Smooth dry concrete sidewalk, flat grade, "
                            "no obstacles detected.",
        unknown_coverage=0.02,
        expected_dominant="instruction",   # κ_I=0.64 > κ_E≈0.025
        expected_action="ASK",
        expected_question_about="instruction",
        description="Instruction has ambiguous target ('there' — no landmark specified). "
                    "Terrain is fully clear (2% GT noise, below 5% threshold). "
                    "Instruction branch dominates (κ_I=0.64 >> κ_E≈0.025). "
                    "Robot asks about target, not terrain.",
    ),
]
