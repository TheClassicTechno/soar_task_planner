# Methodology: Uncertainty-Aware Robot Navigation

> This document defines the general methodology for handling uncertainty in
> outdoor robot navigation. It maps the 6-step pipeline to our code
> implementations, describes the instruction vs. environmental uncertainty
> taxonomy, analyzes baseline limitations, and specifies evaluation metrics
> for each innovation.

---

## 1. Problem Statement

Autonomous outdoor robots navigating in unstructured environments face two
distinct classes of uncertainty:

| Type | Trigger | What is uncertain | Example |
|------|---------|-------------------|---------|
| **Instruction uncertainty** | User gives a movement command | The command itself is ambiguous, incomplete, or contextually unclear | "Keep going" on a trail with a wet grass patch ahead |
| **Environmental uncertainty** | Robot's own perception (no user command) | Whether a terrain region is traversable or safe | Robot sees an unrecognized surface during autonomous navigation |

These two types are **decoupled** and handled by separate system components.
Instruction uncertainty requires a user utterance to trigger. Environmental
uncertainty is detected autonomously by the robot's perception pipeline.

Many existing methods (KnowNo, IntroPlan, WhenToAsk/UPS) address instruction
uncertainty only — they consume a text instruction and decide whether to act
or ask. **Our core innovation is extending this to environmental uncertainty**:
the robot should also ask when it perceives an unknown terrain region ahead,
even if no user instruction has been issued.

---

## 2. The 6-Step Uncertainty Resolution Pipeline

The robot resolves any uncertainty by following these steps in order.
Both instruction and environmental branches enter the pipeline at Step 1 but
diverge at Step 2.

```
     User instruction         Robot perception
          │                         │
          ▼                         ▼
 ┌─────────────────────────────────────────┐
 │  Step 1: Detect uncertainty             │
 │  Instruction: parse command for gaps    │
 │  Environmental: find unknown regions    │
 └─────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────┐
 │  Step 2: Recognize what is unknown      │
 │  Instruction: missing object/action/dir │
 │  Environmental: unknown terrain type,   │
 │    occluded region, unlabeled surface   │
 └─────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────┐
 │  Step 3: Correlate to robot actions     │
 │  Which candidate trajectories/actions   │
 │  are affected by this uncertainty?      │
 └─────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────┐
 │  Step 4: Convert uncertainty to language│
 │  Generate a question that captures the  │
 │  specific missing information.          │
 │  Uses LLM or template bank.             │
 └─────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────┐
 │  Step 5: Update understanding           │
 │  After user response, update the        │
 │  traversability map or intent model.    │
 └─────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────┐
 │  Step 6: Execute selected trajectory    │
 │  Pick the trajectory with highest       │
 │  traversability through confirmed areas │
 └─────────────────────────────────────────┘
```

### Step-to-Code Mapping

| Step | Instruction uncertainty | Environmental uncertainty |
|------|------------------------|--------------------------|
| 1. Detect | `IntroPlanRunner.run_scenario()` / `WhenToAskRunner.run_scenario()` | `EnvironmentalUncertaintyDetector.detect()` |
| 2. Recognize | LLM via `format_introspective_predict_prompt()` / intent factorization | SAM3 + SAM2 spatial subtraction |
| 3. Correlate to actions | `ConformalPredictor.predict_set()` → action options A/B/C/D | `TrajectoryGenerator.score_trajectory()` |
| 4. Convert to language | `WhenToAskRunner` → "ASK" decision | `QuestionGenerator.generate()` |
| 5. Update understanding | — (single-turn; future work: multi-turn) | `MapUpdater.apply_feedback()` |
| 6. Execute | `robot_decision` = "A"/"B"/"C"/"D" | `TrajectoryGenerator.select_best_trajectory()` |

---

## 3. Instruction Uncertainty Taxonomy

Instruction uncertainty arises when the user issues a command that lacks
enough information for the robot to act safely and correctly.

### Sub-types (6 classes)

| Sub-type | Description | Example instruction | Missing info | Our dataset entries |
|----------|-------------|--------------------|-----------|--------------------|
| **Missing object** | Command references an unstated destination | "Go there" | What is "there"? | Type 1 entries |
| **Missing action** | A noun is given but no verb | "The bench" | Open, go to, or stop at? | Type 1 entries |
| **Missing direction** | Directional command without orientation | "Turn" | Left or right? | Type 1 entries |
| **Missing distance** | Vague spatial modifier | "Move forward a bit" | How many meters? | Type 1 entries |
| **Ambiguous target** | Multiple candidates match the description | "Go to the plastic one" (3 objects) | Which plastic object? | Type 1 entries |
| **Ambiguous action** | Multiple valid interpretations | "Go left" (two left paths) | Which left path? | Type 1 entries |

### Terrain-contextual instruction uncertainty (Type 2)

A special sub-type where the instruction is grammatically clear but terrain
conditions create a decision problem: the robot knows what to do in general
but needs to decide between proceeding autonomously or asking.

Example: "Continue through the park" when the forward path has wet grass.
The instruction is unambiguous; the uncertainty is about traversability.

These are our **Type 2 dataset entries** (26 of 57 calibration, 17 of 30 test).
They are tested by IntroPlan and WhenToAsk baselines.

---

## 4. Environmental Uncertainty Definition

Environmental uncertainty is **autonomous and perception-driven**. The robot
detects it by observing the world — no user instruction is required.

### When environmental uncertainty occurs

The robot enters the environmental uncertainty state when its perception
pipeline identifies a region of the scene that it **cannot classify with
sufficient confidence**. This happens when:

1. The region contains a terrain type not in SAM3's 13-class vocabulary
2. The region is partially occluded and depth cannot be inferred
3. SAM2 (general segmentation) detects a region that SAM3 does not label

### Spatial Subtraction Algorithm (Core Innovation)

```
image
  │
  ├── SAM3 (text-grounded segmentation, 13 terrain classes)
  │     └── known_coverage: (H, W) bool mask of all labeled regions
  │
  └── SAM2 (segment everything, no text prompt)
        └── all_regions: list of (mask, score) for every detected region
                │
                └── For each region in all_regions:
                      overlap = |region ∩ known_coverage| / |region|
                      if overlap < 0.3:
                          → UNKNOWN region (not explained by SAM3)
```

**Overlap threshold** = 0.3: a SAM2 region is "unknown" if fewer than 30%
of its pixels are covered by SAM3's known-class masks.

**Traversability scoring**:

| SAM3 class | Traversability score | Rationale |
|-----------|---------------------|-----------|
| sidewalk, road, concrete | 0.95 | Confirmed safe surface |
| grass | 0.90 | Generally walkable |
| dirt | 0.80 | Rough but traversable |
| gravel | 0.70 | Stable enough |
| mulch, sand | 0.65 | Soft but passable |
| vegetation | 0.60 | Possible obstacle |
| wet surface | 0.40 | Slippery risk |
| cracked pavement | 0.35 | Tripping hazard |
| slope | 0.30 | Incline risk |
| rock-bed | 0.20 | Unstable surface |
| mud | 0.10 | Near-impassable |
| water, puddle | 0.05 | Do not cross |
| tree, log, person | 0.05 | Do not cross |
| **unknown** | **0.00** | No information — treat as impassable until clarified |

### Decision rule

After scoring all trajectory waypoints against the traversability map:

- If the **best available trajectory** passes through ≥1 UNKNOWN region:
  → `robot_action = "ASK"` (robot generates a clarification question)
- If all candidate trajectories can be planned through KNOWN regions:
  → `robot_action = "PROCEED"` (robot selects highest-traversability trajectory)
- If the **minimum traversability** along the best path < 0.2:
  → `robot_action = "STOP"` (robot cannot safely continue)

---

## 5. Baseline Gap Analysis

We evaluate three existing baselines on our instruction-uncertainty test set
(30 scenarios, 17 correct = "B" / ask-human).

### Results (nav_test.json, 30 scenarios)

| Metric | IntroPlan | WhenToAsk (UPS) | SAM3 | Gap / Insight |
|--------|-----------|----------------|------|---------------|
| SR (Success Rate) | 0.767 | 0.800 | N/A* | WhenToAsk higher because it asks more |
| HR (Human Help %) | 0.000 | 0.767 | N/A* | IntroPlan never asks; WhenToAsk almost always does |
| FPR (Over-asking %) | 0.000 | 0.348 | N/A* | WhenToAsk asks when it shouldn't 35% of the time |
| NCR (Non-compliant) | 0.233 | 0.967 | N/A* | WhenToAsk sets are very wide (tau=0.86) |
| ESR (Exact Set Rate) | 0.767 | 0.033 | N/A* | IntroPlan makes precise single-option sets |
| Calibrated tau | 0.150 | 0.862 | — | WhenToAsk calibrates very conservatively |
| LLM calls per scenario | 1 | 4 (factorized) | 13 (one per concept) | WhenToAsk: 5× cost |
| Terrain perception | None | None | mIoU=0.167 | SAM3 perceives terrain but output disconnected |

*SAM3 is a perception baseline (segmentation mIoU), not a decision-making baseline.
It does not output ask/act decisions, so SR/HR/FPR/NCR/ESR do not apply.

### Key gaps these results expose

**Gap 1 — IntroPlan never asks (HR=0%)**
IntroPlan uses retrieval-augmented conformal prediction. Its calibrated tau is
very low (0.15 = alpha), producing tight prediction sets almost always of size 1.
It almost never defers to the human. This is a miscalibration problem: the
retrieval makes the LLM overconfident, and the 57-scenario calibration set
may not capture the full distribution of hard cases.

**Gap 2 — WhenToAsk over-asks (FPR=35%)**
WhenToAsk's Bayesian intent factorization spreads probability mass across
many options, making the model more uncertain. Combined with a large tau
(0.86) calibrated from 57 scenarios, it nearly always CLARIFYs. It asks when
it doesn't need to in 35% of cases, which degrades user experience.

**Gap 3 — SAM3 perception is disconnected**
SAM3 segments terrain with 16.7% mIoU (good on grass/vegetation, 0% on rare
classes). But the segmentation output is never used in the ask/act decision.
There is no mechanism connecting terrain perception to navigation decisions.

**The innovation addresses Gap 3**: build the missing connection between
perception (SAM3+SAM2) and the ask/act decision for environmental scenarios.

---

## 6. Evaluation Design

### 6.1 Instruction uncertainty metrics (existing, applies to all 3 baselines)

| Metric | Definition | Expected range |
|--------|-----------|----------------|
| SR | Fraction of scenarios where robot_decision == correct_option or (correct_option=="B" and robot asked) | [0, 1] higher better |
| HR | Fraction of scenarios where robot asked the human | [0, 1] should be ≈ fraction with correct_option=="B" |
| FPR | Fraction of "asked" scenarios where asking was unnecessary | [0, 1] lower better |
| NCR | Fraction where correct option not in prediction set | [0, 1] lower better |
| ESR | Fraction where prediction set == {correct_option} exactly | [0, 1] higher better |

### 6.2 Environmental uncertainty metrics (new, for innovation evaluation)

These apply to nav_env_test.json (instruction=null scenarios) and the
`EnvironmentalUncertaintyRunner`.

| Metric | Symbol | Definition | Expected range |
|--------|--------|-----------|----------------|
| Unknown Region Detection Rate | URDR | True positive rate for detecting unknown regions | [0, 1] higher better |
| Safe Proceed Identification Rate | SPIR | Fraction of "PROCEED" scenarios correctly identified as safe | [0, 1] higher better |
| Appropriate Ask Rate | AAR | Fraction of "ASK" scenarios where robot asked | [0, 1] higher better |
| Spurious Ask Rate | SAR | Fraction of "PROCEED" scenarios where robot incorrectly asked | [0, 1] lower better |
| Question Relevance Score | QRS | Human evaluation (1–5) of generated question quality | [1, 5] higher better |
| Map Update Accuracy | MUA | Fraction of traversability updates correctly applied | [0, 1] higher better |

### 6.3 Qualitative evaluation

Qualitative results demonstrate:
1. **Detection visualization**: side-by-side SAM3 (known) vs SAM2 (all) with
   unknown regions highlighted in red
2. **Trajectory scoring**: color-coded traversability heatmap with candidate
   trajectories overlaid
3. **Question generation examples**: robot question vs ground-truth question_template
4. **Map update sequence**: before/after traversability map on 3 example scenes
5. **Failure analysis**: scenarios where the system asks incorrectly, with explanation

---

## 7. System Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │         Robot Navigation System          │
                    └─────────────────────────────────────────┘
                                        │
           ┌────────────────────────────┴──────────────────────┐
           │                                                     │
           ▼                                                     ▼
 ┌─────────────────┐                               ┌──────────────────────┐
 │  INSTRUCTION    │                               │  ENVIRONMENTAL       │
 │  UNCERTAINTY    │                               │  UNCERTAINTY         │
 │                 │                               │                      │
 │  IntroPlan      │                               │  detector.py         │
 │  WhenToAsk/UPS  │                               │  ├─ SAM3 (known)     │
 │                 │                               │  └─ SAM2 (unknown)   │
 │  Input:         │                               │                      │
 │    instruction  │                               │  traversability.py   │
 │    terrain desc │                               │  trajectory.py       │
 │                 │                               │  question_generator  │
 │  Output:        │                               │  map_updater.py      │
 │    A/B/C/D/ASK  │                               │                      │
 └─────────────────┘                               │  Input:  image only  │
                                                   │  Output: ASK/PROCEED │
                                                   └──────────────────────┘
```

### Code locations

| Component | File | Description |
|-----------|------|-------------|
| Instruction baselines | `baselines/introplan/` | IntroPlan (retrieval + conformal) |
| | `baselines/whentoask/` | WhenToAsk/UPS (intent factorization) |
| | `baselines/sam3/` | SAM3 terrain segmentation (perception baseline) |
| Environmental system | `system/env_uncertainty/detector.py` | SAM3+SAM2 spatial subtraction |
| | `system/env_uncertainty/traversability.py` | Traversability scoring map |
| | `system/env_uncertainty/trajectory.py` | Candidate trajectory scoring |
| | `system/env_uncertainty/question_generator.py` | Template + LLM question generation |
| | `system/env_uncertainty/map_updater.py` | Traversability map update after feedback |
| | `system/env_uncertainty/runner.py` | End-to-end runner for env scenarios |
| Datasets | `baselines/introplan/data/nav_calibration.json` | 57 instruction-uncertainty calibration |
| | `baselines/introplan/data/nav_test.json` | 30 instruction-uncertainty test |
| | `baselines/introplan/data/nav_env_test.json` | 20 environmental uncertainty test |
