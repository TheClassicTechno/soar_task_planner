"""
Generate additional IntroPlan calibration scenarios using Claude Vision.

This script:
  1. Loads RUGD images from the downloaded dataset
  2. Sends each image to Claude Vision to get a terrain description
  3. Assigns a random navigation instruction from our 5 uncertainty types
  4. Asks Claude to generate the correct option + reasoning
  5. Saves the resulting scenarios to data/nav_generated.json

The generated scenarios supplement the 20 hand-crafted ones in nav_calibration.json.
For calibration we need ~50 total; for richer knowledge base, generate up to 200.

Usage:
    python -m baselines.introplan.generate_calibration_data \
        --rugd_dir /path/to/rugd \
        --n_images 30 \
        --output baselines/introplan/data/nav_generated.json
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from baselines.introplan.llm_interface import LLMInterface
from baselines.introplan.navigation_prompts import (
    KNOWLEDGE_BASE_ENTRY_PROMPT,
    OPTION_DESCRIPTIONS,
    SCENE_DESCRIPTION_PROMPT,
    UNCERTAINTY_TYPE_LABELS,
    format_kb_entry_prompt,
)
from baselines.sam3.data_loader import RUGDSample, load_rugd_split


# ── Instruction templates per uncertainty type ────────────────────────────────
# Each template is a realistic navigation command. The LLM will determine the
# correct robot response given the terrain observation.

INSTRUCTION_TEMPLATES: Dict[int, List[str]] = {
    1: [  # Instructional ambiguity — vague reference or unclear destination
        "Go that way",
        "Take me there",
        "Continue like before",
        "Head toward the thing I mentioned",
        "Let's keep going",
        "Take me the usual way",
        "Go to the place we talked about",
    ],
    2: [  # Environmental uncertainty — terrain seen, user preference unknown.
        # Use movement commands so the scenario is consistent with RUGD terrain images,
        # which show outdoor terrain rather than named buildings or destinations.
        "Keep going",
        "Continue forward",
        "Go straight",
        "Move ahead",
        "Don't slow down",
        "Take the shortcut",
        "Keep moving",
    ],
    3: [  # Safety critical — immediate hazard, robot must act conservatively
        "Keep going straight",
        "Don't slow down",
        "Just move forward",
        "Continue on this path",
        "Ignore the detour signs",
        "Hurry up, just go",
    ],
    # Type 4 (system/perception error) is NOT generated from RUGD images —
    # RUGD captures terrain, not robot system state. Type 4 entries must be
    # hand-crafted. See nav_calibration.json entries nav_051–nav_058.
}


LABEL_GENERATION_PROMPT = """You are an expert robot navigation planner reviewing this situation:

Instruction: "{instruction}"
Terrain observation: "{terrain_description}"
Uncertainty type: {uncertainty_type_label}

The robot has 4 candidate responses:
  A: {option_a}
  B: {option_b}
  C: {option_c}
  D: {option_d}

Which single option is MOST appropriate for this situation, and why?
Consider: Is there a safety hazard? Is user preference unknown? Is the instruction ambiguous?

Respond in JSON format:
{{
  "correct_option": "<A, B, C, or D>",
  "reasoning": "<2-3 sentences explaining why this option is correct>"
}}"""


def generate_scenario(
    llm: LLMInterface,
    sample: RUGDSample,
    uncertainty_type: int,
    scenario_id: str,
) -> Optional[Dict]:
    """
    Generate one labeled scenario from a RUGD image.

    Returns None if the image fails to load or LLM returns invalid JSON.
    """
    try:
        image = sample.load_image()
    except Exception:
        return None

    # Step 1: Get terrain description from Claude Vision
    terrain_description = llm.describe_image(image, SCENE_DESCRIPTION_PROMPT).strip()
    if not terrain_description or len(terrain_description) < 10:
        return None

    # Step 2: Pick a random instruction for this uncertainty type
    instruction = random.choice(INSTRUCTION_TEMPLATES[uncertainty_type])

    # Step 3: Build option set (reuse defaults — specific question in B generated below)
    options = {
        "A": "Continue on the current path",
        "B": f"Ask user a targeted question about the terrain: {terrain_description[:50]}...",
        "C": "Reroute automatically to avoid the uncertain terrain",
        "D": "Slow down and proceed cautiously",
    }

    # Step 4: Ask Claude to label the correct option
    uncertainty_label = UNCERTAINTY_TYPE_LABELS.get(uncertainty_type, f"Type {uncertainty_type}")
    label_prompt = LABEL_GENERATION_PROMPT.format(
        instruction=instruction,
        terrain_description=terrain_description,
        uncertainty_type_label=uncertainty_label,
        option_a=options["A"],
        option_b=options["B"],
        option_c=options["C"],
        option_d=options["D"],
    )

    try:
        label_result = llm.predict_json(label_prompt)
    except Exception:
        return None

    correct_option = label_result.get("correct_option", "").strip().upper()
    if correct_option not in ["A", "B", "C", "D"]:
        return None

    reasoning = label_result.get("reasoning", "")

    return {
        "entry_id": scenario_id,
        "instruction": instruction,
        "terrain_description": terrain_description,
        "uncertainty_type": uncertainty_type,
        "options": options,
        "correct_option": correct_option,
        "reasoning": reasoning,
        "source_image": str(sample.name),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate IntroPlan calibration data from RUGD")
    p.add_argument(
        "--rugd_dir", default=None,
        help="Root RUGD directory. Overrides RUGD_DATA_PATH env var.",
    )
    p.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
        help="RUGD split to sample images from",
    )
    p.add_argument(
        "--n_images", type=int, default=30,
        help="Number of images to process (each becomes one scenario)",
    )
    p.add_argument(
        "--output", default="baselines/introplan/data/nav_generated.json",
        help="Output JSON file path",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible instruction assignment",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    random.seed(args.seed)

    rugd_dir = args.rugd_dir or os.environ.get("RUGD_DATA_PATH")
    if not rugd_dir:
        raise ValueError("Provide --rugd_dir or set RUGD_DATA_PATH in .env")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if anthropic_key:
        api_key, api_type, model = anthropic_key, "anthropic", "claude-sonnet-4-6"
    elif openai_key:
        api_key, api_type, model = openai_key, "openai", "gpt-4o"
    else:
        raise ValueError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file")

    print(f"Using {api_type} ({model})")
    print(f"Loading RUGD {args.split} split from: {rugd_dir}")
    all_samples = load_rugd_split(rugd_dir, split=args.split)

    # Subsample evenly
    step = max(1, len(all_samples) // args.n_images)
    samples = all_samples[::step][: args.n_images]
    print(f"Processing {len(samples)} images (sampled from {len(all_samples)} total)")

    llm = LLMInterface(api_key=api_key, api_type=api_type, model=model)
    scenarios = []

    # Cycle through types 1/2/3 only — Type 4 (system error) cannot be
    # derived from RUGD images and must remain hand-crafted.
    uncertainty_types = [1, 2, 3]

    for idx, sample in enumerate(tqdm(samples, desc="Generating scenarios")):
        utype = uncertainty_types[idx % len(uncertainty_types)]
        scenario_id = f"gen_{idx:04d}"

        result = generate_scenario(llm, sample, utype, scenario_id)
        if result:
            scenarios.append(result)

    print(f"\nGenerated {len(scenarios)} scenarios (from {len(samples)} images)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    # Summary stats
    from collections import Counter
    type_counts = Counter(s["uncertainty_type"] for s in scenarios)
    option_counts = Counter(s["correct_option"] for s in scenarios)
    print(f"\nUncertainty type distribution: {dict(type_counts)}")
    print(f"Correct option distribution:   {dict(option_counts)}")
    print(f"\nSaved to: {out_path}")
    print(f"Total {api_type} API calls: {llm.total_calls}")


if __name__ == "__main__":
    main()
