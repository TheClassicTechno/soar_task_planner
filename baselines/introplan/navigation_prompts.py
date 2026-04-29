"""
Prompt templates for the IntroPlan navigation baseline.

Adapted from Liang et al. NeurIPS 2024 "Introspective Planning" (Mobile
Manipulation domain) to the outdoor robot navigation under uncertainty domain.

The key adaptation:
  - Original: pick/place object tasks with option sets like {A:grab mug, B:ask where mug is}
  - Ours: navigation decisions with option sets like {A:continue, B:ask user, C:reroute, D:slow down}

Three prompt types are used in the IntroPlan pipeline:
  1. CANDIDATE_OPTIONS_PROMPT  — given scene + instruction → generate 4 candidate options
  2. KNOWLEDGE_BASE_PROMPT     — given instruction + correct option → generate introspective
                                  reasoning example (offline, during KB construction)
  3. INTROSPECTIVE_PREDICT_PROMPT — given scene + instruction + retrieved KB examples →
                                    reason about each option → predict best option + confidence
"""

from typing import Dict, List


# ── Option labels ─────────────────────────────────────────────────────────────
# We fix 4 options per scenario. Option A always means "act (proceed)",
# options B/C/D are clarification or conservative alternatives.
# This keeps the conformal prediction calibration consistent.

OPTION_LABELS = ["A", "B", "C", "D"]

OPTION_DESCRIPTIONS = {
    "A": "Continue on the current path (proceed without asking)",
    "B": "Ask the user a targeted clarifying question about the terrain",
    "C": "Reroute automatically to avoid the uncertain terrain",
    "D": "Slow down and proceed cautiously without asking",
}


# ── 1. Candidate options prompt ───────────────────────────────────────────────

CANDIDATE_OPTIONS_PROMPT = """You are a navigation assistant for a robot helping a person outdoors.

Current situation:
  Instruction from user: "{instruction}"
  Terrain observation:   "{terrain_description}"
  Uncertainty type:      {uncertainty_type_label}

Generate exactly 4 candidate robot responses labeled A, B, C, D:
  A: Continue on the current path (proceed without asking)
  B: Ask the user a targeted clarifying question specific to this terrain
  C: Reroute automatically to avoid the uncertain terrain
  D: Slow down and proceed cautiously without asking

For option B, write the exact question the robot should ask (1 sentence, specific to the observed terrain feature).

Respond in JSON format:
{{
  "A": "Continue on the current path",
  "B": "<specific clarifying question>",
  "C": "Reroute automatically to avoid <terrain feature>",
  "D": "Slow down and proceed cautiously through <terrain feature>"
}}"""


# ── 2. Knowledge base entry construction prompt ───────────────────────────────
# Used offline to build the retrieval knowledge base from labeled examples.

KNOWLEDGE_BASE_ENTRY_PROMPT = """You are an expert robot navigation planner.

A robot received this situation:
  Instruction: "{instruction}"
  Terrain:     "{terrain_description}"
  Uncertainty: {uncertainty_type_label}

The human expert chose option {correct_option} as the best response:
  {correct_option}: {correct_option_text}

Write a brief introspective reasoning entry (3-4 sentences) explaining WHY option {correct_option}
is correct here. Focus on:
  1. What specific feature makes this uncertain
  2. Why proceeding without asking would be risky OR why asking is unnecessary overhead
  3. The key decision factor (user preference unknown? safety critical? instruction ambiguous?)

Keep it concise and generalizable — this entry will be retrieved to help future similar situations.

Reasoning:"""


# ── 3. Introspective prediction prompt ────────────────────────────────────────
# Used at inference: given new scenario + retrieved KB examples → reason → predict.

INTROSPECTIVE_PREDICT_PROMPT = """You are a navigation assistant for a robot helping a person outdoors.
Your task is to decide the best response for the current situation.

=== RETRIEVED EXAMPLES FROM PAST SITUATIONS ===
{retrieved_examples}

=== CURRENT SITUATION ===
Instruction from user: "{instruction}"
Terrain observation:   "{terrain_description}"

=== CANDIDATE OPTIONS ===
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}

=== INSTRUCTIONS ===
Using the examples above as guidance, reason through each option step by step.
Then predict which single option is most appropriate.

Consider:
- Is this terrain safe to traverse without asking? (Type 3 = act conservatively — never ask for permission on a safety hazard)
- Is the robot's own sensor/system state reliable? (Type 4 = stop or alert user if system/sensor error, do not proceed blindly)
- Is the instruction itself unclear? (Type 1 = ask for clarification before moving)
- Would rerouting or slowing down resolve the uncertainty without bothering the user?

Respond in this exact JSON format:
{{
  "reasoning": {{
    "A": "<1-2 sentence analysis of option A>",
    "B": "<1-2 sentence analysis of option B>",
    "C": "<1-2 sentence analysis of option C>",
    "D": "<1-2 sentence analysis of option D>"
  }},
  "prediction": "<single letter: A, B, C, or D>",
  "confidence": <float between 0.0 and 1.0>,
  "explanation": "<1 sentence: why this option is best>"
}}"""


# ── 4. Scene description prompt (for calibration data generation) ─────────────
# Used by generate_calibration_data.py to describe a RUGD image.

SCENE_DESCRIPTION_PROMPT = """Describe the terrain features visible in this outdoor scene.
Focus only on what is relevant to robot navigation safety and surface quality.

List the most salient terrain features (maximum 3), using this format:
"<terrain feature> detected <approximate distance>, confidence <high/medium/low>"

Examples of relevant features: cracked pavement, gravel, puddle, wet surface,
steep slope, curb, mud, overgrown vegetation, narrow path, smooth asphalt.

Be specific and concise. One line per feature."""


# ── Helper functions ──────────────────────────────────────────────────────────

# Type taxonomy finalized April 2026 (see project_type_taxonomy.md).
# Type 1: instructional ambiguity — user command vague or incomplete.
# Type 2: terrain/environmental — robot sees terrain; user preference unknown.
# Type 3: safety critical — immediate hazard; robot must act autonomously, never ask.
# Type 4: system/perception error — sensor, localization, or planner unreliable;
#         robot must stop or alert user before acting.
# Type 4 entries cannot be auto-generated from RUGD images (terrain ≠ system state).
UNCERTAINTY_TYPE_LABELS = {
    1: "Type 1 (instructional ambiguity — user command is vague, incomplete, or has no clear referent)",
    2: "Type 2 (terrain/environmental uncertainty — robot sees a terrain feature; user preference or safety tolerance for that terrain is unknown)",
    3: "Type 3 (safety critical — immediate hazard detected; robot must act conservatively without asking)",
    4: "Type 4 (system/perception error — robot sensor, localization, or planning subsystem is unreliable; robot must stop or alert user before acting)",
}


def format_candidate_options_prompt(
    instruction: str,
    terrain_description: str,
    uncertainty_type: int,
) -> str:
    """Fill in the candidate options prompt template."""
    label = UNCERTAINTY_TYPE_LABELS.get(uncertainty_type, f"Type {uncertainty_type}")
    return CANDIDATE_OPTIONS_PROMPT.format(
        instruction=instruction,
        terrain_description=terrain_description,
        uncertainty_type_label=label,
    )


def format_kb_entry_prompt(
    instruction: str,
    terrain_description: str,
    uncertainty_type: int,
    correct_option: str,
    correct_option_text: str,
) -> str:
    """Fill in the KB entry construction prompt."""
    label = UNCERTAINTY_TYPE_LABELS.get(uncertainty_type, f"Type {uncertainty_type}")
    return KNOWLEDGE_BASE_ENTRY_PROMPT.format(
        instruction=instruction,
        terrain_description=terrain_description,
        uncertainty_type_label=label,
        correct_option=correct_option,
        correct_option_text=correct_option_text,
    )


def format_retrieval_examples(examples: List[Dict]) -> str:
    """
    Format a list of KB entries into the retrieved examples block
    for the introspective prediction prompt.

    Each example dict should have: instruction, terrain_description, reasoning, correct_option.
    """
    if not examples:
        return "(No similar past examples found)"

    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Instruction: {ex.get('instruction', '')}")
        lines.append(f"  Terrain: {ex.get('terrain_description', '')}")
        lines.append(f"  Correct response: Option {ex.get('correct_option', '?')}")
        lines.append(f"  Reasoning: {ex.get('reasoning', '')}")
        lines.append("")
    return "\n".join(lines).strip()


def format_introspective_predict_prompt(
    instruction: str,
    terrain_description: str,
    options: Dict[str, str],
    retrieved_examples: List[Dict],
) -> str:
    """Fill in the introspective prediction prompt."""
    return INTROSPECTIVE_PREDICT_PROMPT.format(
        instruction=instruction,
        terrain_description=terrain_description,
        option_a=options.get("A", OPTION_DESCRIPTIONS["A"]),
        option_b=options.get("B", OPTION_DESCRIPTIONS["B"]),
        option_c=options.get("C", OPTION_DESCRIPTIONS["C"]),
        option_d=options.get("D", OPTION_DESCRIPTIONS["D"]),
        retrieved_examples=format_retrieval_examples(retrieved_examples),
    )
