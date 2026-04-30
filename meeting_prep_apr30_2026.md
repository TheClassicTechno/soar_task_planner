# Phase A + B Progress — Meeting Prep (April 30, 2026)

**What this covers:** Changes made after the April 27 meeting to fix dataset balance, the FPR metric, and stale code comments. Also includes the scenario table and pipeline overview the mentor asked for.

---

## What the April 27 Meeting Asked For

The team identified these problems that needed fixing before the next meeting:

1. The test dataset had **zero cases where the robot should just proceed** — making FPR trivially 0%
2. **70% of test answers were "ask user"** — a baseline that always asks would score 70% accuracy
3. The **false positive metric was too narrow** — it only counted cases where option A was correct, missing cases where the robot should reroute or stop autonomously (options C and D)
4. The dataset needed a **`should_ask` field** to clearly mark when the robot should ask vs. act
5. **Stale comments** in the code still described the old taxonomy (Type 4 as "terrain preference")

---

## Phase A — What Was Fixed

### 1. `baselines/introplan/metrics.py` — FPR definition

**Old definition (broken):**
> False positive = robot asked when correct option was A (proceed directly)

This only caught one narrow case and missed two others.

**New definition (meeting-aligned):**
> False positive = robot asked when the correct action was to act autonomously — options A (proceed), C (reroute), or D (slow down). Option B is the only action that involves asking the user.

Code change:
```python
# Before
return self.asked_human and self.correct_option == "A"

# After
return self.asked_human and self.correct_option != "B"
```

**Why this matters:** With the old definition, FPR was always 0% even when the robot asked unnecessarily on safety-critical scenarios where it should have just stopped or rerouted. The new definition catches all cases where the robot over-asks.

---

### 2. `baselines/introplan/navigation_prompts.py` — Stale taxonomy comments

Two places in the file still described the old Type 4 taxonomy ("ask if unknown" for terrain preference). Both were updated to match the finalized taxonomy.

**Fixed in the prediction prompt:**
```
# Before
(Type 4 = ask if unknown)

# After
(Type 4 = stop or alert user if system/sensor error, do not proceed blindly)
```

**Fixed in the type label comment block:**
```python
# Before (stale — referenced an unresolved merge of Type 4 into Type 2)
# Type 4 covers mild terrain where preference is unlikely to matter but is still
# unrecorded. The meeting discussed folding Type 4 into Type 2 — keep separate
# for now to allow baseline ablation...

# After (matches finalized taxonomy)
# Type 4: system/perception error — sensor, localization, or planner unreliable;
#         robot must stop or alert user before acting.
# Type 4 entries cannot be auto-generated from RUGD images (terrain ≠ system state).
```

---

### 3. `baselines/introplan/data/nav_calibration.json` — Added `should_ask` field

Added `should_ask: true/false` to all 57 calibration entries. The rule is:
- `should_ask = true` when `correct_option == "B"` (robot should ask the user)
- `should_ask = false` when `correct_option` is A, C, or D (robot should act directly)

| Option | Meaning | should_ask | Count in calibration |
|--------|---------|-----------|---------------------|
| A | Proceed without asking | false | 9 |
| B | Ask the user | **true** | 32 |
| C | Reroute automatically | false | 9 |
| D | Slow down | false | 7 |

No logic changed — this is a data annotation field only.

---

### 4. `baselines/introplan/data/nav_test.json` — Added `should_ask` + 6 new no-uncertainty entries

**Part 1:** Added `should_ask` field to all 24 existing entries (same rule as calibration).

**Part 2:** Added 6 new entries where `correct_option = A` (proceed directly). These are the cases FPR was missing entirely.

| Entry | Type | Instruction | Terrain | Why proceed? |
|-------|------|-------------|---------|-------------|
| nav_t025 | 1 | "Go to the building on the left" | Two buildings, left one clearly identified | Instruction is unambiguous — no clarification needed |
| nav_t026 | 1 | "Stop here" | Open concrete plaza, no obstacles | Command is clear and location-specific |
| nav_t027 | 2 | "Keep going" | Smooth dry concrete, flat, no hazards | Standard surface — benign terrain, no user preference needed |
| nav_t028 | 2 | "Continue forward" | Well-maintained asphalt, gentle 3° slope | 3-degree slope is within normal parameters |
| nav_t029 | 2 | "Move forward" | Smooth paved trail, gentle curve, wide | Gentle curve on a wide path — no uncertainty |
| nav_t030 | 2 | "Let's go" | Flat concrete plaza, faint decorative markings | Decorative markings present no navigational uncertainty |

**Before and after comparison:**

| Metric | Before | After |
|--------|--------|-------|
| Total test entries | 24 | 30 |
| Option A correct (proceed directly) | 0 | **6** |
| Option B correct (ask user) | 17 | 17 |
| Option C correct (reroute) | 3 | 3 |
| Option D correct (slow down/stop) | 4 | 4 |
| % asking correct answers (B rate) | 70.8% | **56.7%** |
| FPR measurable? | **No** — always 0 | **Yes** — 13 scenarios where asking is wrong |

---

### 5. `baselines/introplan/tests/test_metrics.py` — New FPR test cases

Added three new tests to make sure the updated FPR definition is enforced:

| Test | Scenario | Expected result |
|------|----------|----------------|
| `test_result_false_positive_when_asked_and_correct_was_proceed` | correct=A, robot asks | FP = True |
| `test_result_false_positive_when_asked_and_correct_was_reroute` | correct=C, robot asks | FP = True |
| `test_result_false_positive_when_asked_and_correct_was_slowdown` | correct=D, robot asks | FP = True |
| `test_result_not_false_positive_when_asked_and_correct_was_ask` | correct=B, robot asks | FP = False |

These tests would have caught the broken FPR definition immediately if they had existed before.

---

### 6. `baselines/introplan/tests/test_calibration_integrity.py` — `should_ask` validation

Added `should_ask` to the required fields list, and added a new `TestShouldAskField` class with 4 tests:

| Test | What it checks |
|------|---------------|
| `test_should_ask_is_boolean_in_calibration` | Every calibration entry has `should_ask` as a Python bool |
| `test_should_ask_is_boolean_in_test` | Every test entry has `should_ask` as a Python bool |
| `test_should_ask_true_iff_correct_option_is_b` | `should_ask == (correct_option == "B")` for every single entry — if someone edits correct_option without updating should_ask, this catches it |
| `test_test_set_has_option_a_correct_entries` | Test set must have at least one entry with correct_option=A — prevents FPR from silently becoming 0 again |

**Test suite result:** 156 passed, 0 failed (up from 150 — 6 new tests added).

---

## Phase B — Scenario Table (for Wednesday)

This is what the mentor asked for: each uncertainty type listed with concrete examples.

| Type | Name | # in test | # in calibration | Example situation | What robot observes | Correct robot behavior |
|------|------|-----------|-----------------|-------------------|---------------------|----------------------|
| No uncert. | Proceed directly | 6 | 9 | "Keep going" on clear flat asphalt | Smooth dry path, no hazards | **A: Proceed** — no user input needed, act directly |
| 1 | Instruction ambiguity | 7 | 12 | "Go there" at a 3-way fork | Three equally valid paths | **B: Ask user** — "there" is ambiguous without more context |
| 2 | Terrain / environmental | 14 | 26 | "Keep going" near wet leaves | Leaf-covered path with hidden roots | **B: Ask user** — robot sees risky terrain but doesn't know user's preference |
| 3 | Safety critical | 5 | 11 | "Go straight" near a steep drop | Drop-off edge detected ahead | **C/D: Stop or reroute** — hazard is clear; never ask permission before stopping |
| 4 | System / perception error | 4 | 8 | "Continue forward" with GPS lost | Dead-reckoning drift, lost position | **B: Alert user** — robot can't trust its own sensors; must not act blindly |

**Key design rule that came from the meeting:**
- Types 2, 3, 4 may look similar (all involve the robot observing something) but the **correct response is different for each**:
  - Type 2: environment is uncertain → ask user preference
  - Type 3: environment is dangerous → act autonomously, no asking
  - Type 4: robot's own systems are broken → alert user before doing anything

---

## Pipeline — Two-Stage Design (for Wednesday diagram)

The mentor framed the project around two problems. Here is how they map to the pipeline:

```
INPUT
  ├── User instruction: "Go there"
  └── Environment: terrain image / system state

          │
          ▼
┌─────────────────────────────┐
│  STAGE 1: Detect uncertainty │
│  • LLM → parse instruction   │  Does a clear uncertainty exist?
│  • SAM3 → scan terrain       │  If yes, what type (1/2/3/4)?
└─────────────────────────────┘
          │
    ┌─────┴──────────┐
    │                │
  None           Uncertainty
  detected       detected
    │                │
    ▼                ▼
Act directly     Is it Type 3?
(option A)         │
                ┌──┴──┐
               Yes    No (Type 1, 2, or 4)
                │         │
                ▼         ▼
           Stop/reroute  ┌──────────────────────────┐
           autonomously  │ STAGE 2: Clarify          │
           (options C/D) │ • Fixed question bank, OR │
                         │ • LLM-generated question  │
                         └──────────────────────────┘
                                    │
                                    ▼
                          Ask user a specific question
                          (e.g., "Should I cross the wet leaves?")
                                    │
                                    ▼
                          Receive user answer
                                    │
                                    ▼
                          Final robot action
                                    │
                                    ▼
            ┌──────────────────────────────────────────────┐
            │ EVALUATE                                      │
            │ 1. Detection accuracy — did it detect right? │
            │ 2. Question quality — did it ask the right Q? │
            │ 3. Final action accuracy — right outcome?    │
            └──────────────────────────────────────────────┘
```

---

## Open Questions for the Wednesday Meeting

These were not resolved in the April 27 meeting and need a team decision before moving forward:

| Question | Current state | Options | Blocks |
|----------|--------------|---------|--------|
| 3 types or 4 types? | Code has 4 types. Meeting leaned toward 3 (merge safety + system error). | Keep 4 (clearer response logic) vs. simplify to 3 | Taxonomy used in all new entries |
| Full dataset format redesign? | Dataset has action options only. Meeting proposed adding candidate clarification questions, user answer, and final action fields. | Add now (rewrites 81 entries) vs. after baselines | Two-stage evaluation |
| Fixed question bank vs. LLM-generated questions? | Not decided | Fixed bank (easier to evaluate) vs. LLM-generated (more flexible, innovation area) | Stage 2 clarification evaluation |
| KnowNo baseline — when to implement? | Not yet started | After Wednesday alignment | Comparison baseline |

---

## Files Changed

| File | What changed |
|------|-------------|
| `baselines/introplan/metrics.py` | FPR: `correct_option == "A"` → `correct_option != "B"` |
| `baselines/introplan/navigation_prompts.py` | Fixed Type 3/4 descriptions in prediction prompt; replaced stale taxonomy comment |
| `baselines/introplan/data/nav_calibration.json` | Added `should_ask` field to all 57 entries |
| `baselines/introplan/data/nav_test.json` | Added `should_ask` to 24 existing entries; added 6 new no-uncertainty entries (nav_t025–t030) |
| `baselines/introplan/tests/test_metrics.py` | Added FPR tests for options C and D |
| `baselines/introplan/tests/test_calibration_integrity.py` | Added `should_ask` to required fields; added `TestShouldAskField` class (4 tests) |

**Test suite: 156 passed, 0 failed.**
