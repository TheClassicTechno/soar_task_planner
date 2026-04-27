"""
Calibration and test data integrity checks.

These tests enforce the data contract agreed in the April 2025 meeting:
  - Movement-based instructions (not destination) for Type 2 / 3 / 4 entries
  - Required fields present on every entry
  - Valid uncertainty types (1-4)
  - Valid correct_option (A/B/C/D)
  - source_image field present (null or a filename string)
  - Type 1 entries exempt from instruction-format rules (destination ambiguity IS valid)

Run with:
    pytest baselines/introplan/tests/test_calibration_integrity.py -v
"""

import json
from pathlib import Path
from typing import Dict, List

import pytest

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
CALIBRATION_FILE = DATA_DIR / "nav_calibration.json"
TEST_FILE = DATA_DIR / "nav_test.json"

# ── Constants ─────────────────────────────────────────────────────────────────
REQUIRED_FIELDS = {
    "entry_id",
    "instruction",
    "terrain_description",
    "uncertainty_type",
    "options",
    "correct_option",
    "reasoning",
    "source_image",
}

VALID_OPTIONS = {"A", "B", "C", "D"}
VALID_UNCERTAINTY_TYPES = {1, 2, 3, 4}

# Destination-based keywords that must NOT appear as the full instruction for
# Type 2 / 3 / 4 entries.  Type 1 entries are exempt because destination
# ambiguity ("Take me to the office" with two offices visible) is a valid
# Type 1 instructional ambiguity scenario.
_DESTINATION_PREFIXES = (
    "take me to",
    "lead me to",
    "navigate to",
    "go to",
    "get me to",
    "head to",
    "head toward",
)


def _is_destination_instruction(instruction: str) -> bool:
    """Returns True if the instruction names a specific place to navigate to."""
    lower = instruction.strip().lower()
    return any(lower.startswith(prefix) for prefix in _DESTINATION_PREFIXES)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> List[Dict]:
    assert path.exists(), f"Data file not found: {path}"
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def calibration_entries() -> List[Dict]:
    return _load_json(CALIBRATION_FILE)


@pytest.fixture(scope="module")
def test_entries() -> List[Dict]:
    return _load_json(TEST_FILE)


@pytest.fixture(scope="module")
def all_entries(calibration_entries, test_entries) -> List[Dict]:
    return calibration_entries + test_entries


# ── Required field tests ──────────────────────────────────────────────────────

class TestRequiredFields:
    """Every entry must have all required fields."""

    def test_calibration_has_required_fields(self, calibration_entries):
        for entry in calibration_entries:
            missing = REQUIRED_FIELDS - entry.keys()
            assert not missing, (
                f"Entry {entry.get('entry_id')} missing fields: {missing}"
            )

    def test_test_has_required_fields(self, test_entries):
        for entry in test_entries:
            missing = REQUIRED_FIELDS - entry.keys()
            assert not missing, (
                f"Entry {entry.get('entry_id')} missing fields: {missing}"
            )

    def test_source_image_is_null_or_string(self, all_entries):
        """source_image must be None (synthetic) or a non-empty string (RUGD)."""
        for entry in all_entries:
            val = entry["source_image"]
            assert val is None or (isinstance(val, str) and len(val) > 0), (
                f"Entry {entry['entry_id']}: source_image must be null or non-empty string, got {val!r}"
            )


# ── Uncertainty type tests ────────────────────────────────────────────────────

class TestUncertaintyTypes:
    """All uncertainty_type values must be in {1, 2, 3, 4}."""

    def test_calibration_valid_types(self, calibration_entries):
        for entry in calibration_entries:
            assert entry["uncertainty_type"] in VALID_UNCERTAINTY_TYPES, (
                f"Entry {entry['entry_id']}: invalid uncertainty_type {entry['uncertainty_type']}"
            )

    def test_test_valid_types(self, test_entries):
        for entry in test_entries:
            assert entry["uncertainty_type"] in VALID_UNCERTAINTY_TYPES, (
                f"Entry {entry['entry_id']}: invalid uncertainty_type {entry['uncertainty_type']}"
            )


# ── Correct option tests ──────────────────────────────────────────────────────

class TestCorrectOptions:
    """correct_option must be A, B, C, or D and must exist in options dict."""

    def test_correct_option_is_valid_letter(self, all_entries):
        for entry in all_entries:
            assert entry["correct_option"] in VALID_OPTIONS, (
                f"Entry {entry['entry_id']}: invalid correct_option {entry['correct_option']!r}"
            )

    def test_correct_option_exists_in_options(self, all_entries):
        for entry in all_entries:
            opts = entry["options"]
            co = entry["correct_option"]
            assert co in opts, (
                f"Entry {entry['entry_id']}: correct_option '{co}' not in options keys {list(opts.keys())}"
            )

    def test_options_has_exactly_four_keys(self, all_entries):
        for entry in all_entries:
            assert set(entry["options"].keys()) == VALID_OPTIONS, (
                f"Entry {entry['entry_id']}: options must have keys A/B/C/D, got {list(entry['options'].keys())}"
            )


# ── Instruction format tests ──────────────────────────────────────────────────

class TestInstructionFormat:
    """
    Type 2 / 3 / 4 entries must NOT use destination-based instructions.
    Reason: RUGD images show terrain, not campus buildings.  Destination
    commands introduce a second uncertainty (location) that contaminates the
    intended uncertainty type.  Type 1 is exempt — destination ambiguity is
    a valid Type 1 scenario.
    """

    def _destination_violations(self, entries: List[Dict]) -> List[str]:
        violations = []
        for entry in entries:
            if entry["uncertainty_type"] in {2, 3, 4}:
                if _is_destination_instruction(entry["instruction"]):
                    violations.append(
                        f"{entry['entry_id']} (type={entry['uncertainty_type']}): "
                        f"'{entry['instruction']}'"
                    )
        return violations

    def test_calibration_no_destination_instructions_in_type234(self, calibration_entries):
        violations = self._destination_violations(calibration_entries)
        assert not violations, (
            "Type 2/3/4 entries must use movement commands, not destination commands.\n"
            "Violations:\n" + "\n".join(f"  {v}" for v in violations)
        )

    def test_test_no_destination_instructions_in_type234(self, test_entries):
        violations = self._destination_violations(test_entries)
        assert not violations, (
            "Type 2/3/4 entries must use movement commands, not destination commands.\n"
            "Violations:\n" + "\n".join(f"  {v}" for v in violations)
        )

    def test_instructions_are_non_empty(self, all_entries):
        for entry in all_entries:
            assert entry["instruction"].strip(), (
                f"Entry {entry['entry_id']}: instruction must not be empty"
            )


# ── Reasoning quality tests ───────────────────────────────────────────────────

class TestReasoningQuality:
    """Reasoning must be non-empty and at least minimally informative."""

    def test_reasoning_non_empty(self, all_entries):
        for entry in all_entries:
            assert entry["reasoning"].strip(), (
                f"Entry {entry['entry_id']}: reasoning must not be empty"
            )

    def test_reasoning_mentions_uncertainty_type(self, all_entries):
        """Reasoning should reference the correct type keyword."""
        type_keywords = {
            # "clarif" catches "clarification" in entries that correctly decide NOT to ask
            1: ["type 1", "ambig", "unclear", "vague", "referent", "clarif"],
            # "uncertain" catches "uncertainty" and negation form "not a case of uncertainty"
            2: ["type 2", "environmental", "terrain", "preference", "uncertain"],
            3: ["type 3", "safety", "hazard", "critical", "immediate"],
            4: ["type 4", "system", "error", "gps", "perception", "sensor", "drift", "confidence", "navigation"],
        }
        for entry in all_entries:
            utype = entry["uncertainty_type"]
            keywords = type_keywords.get(utype, [])
            reasoning_lower = entry["reasoning"].lower()
            matched = any(kw in reasoning_lower for kw in keywords)
            assert matched, (
                f"Entry {entry['entry_id']} (type {utype}): reasoning does not mention "
                f"any expected keyword from {keywords}"
            )


# ── Dataset composition tests ─────────────────────────────────────────────────

class TestDatasetComposition:
    """Check that all four uncertainty types are represented in the datasets."""

    def _type_counts(self, entries: List[Dict]) -> Dict[int, int]:
        counts: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        for entry in entries:
            counts[entry["uncertainty_type"]] = counts.get(entry["uncertainty_type"], 0) + 1
        return counts

    def test_calibration_has_all_four_types(self, calibration_entries):
        counts = self._type_counts(calibration_entries)
        for utype in VALID_UNCERTAINTY_TYPES:
            assert counts[utype] > 0, (
                f"Calibration set has zero entries for uncertainty type {utype}"
            )

    def test_test_has_all_four_types(self, test_entries):
        counts = self._type_counts(test_entries)
        for utype in VALID_UNCERTAINTY_TYPES:
            assert counts[utype] > 0, (
                f"Test set has zero entries for uncertainty type {utype}"
            )

    def test_calibration_entry_ids_unique(self, calibration_entries):
        ids = [e["entry_id"] for e in calibration_entries]
        assert len(ids) == len(set(ids)), "Duplicate entry_ids in calibration set"

    def test_test_entry_ids_unique(self, test_entries):
        ids = [e["entry_id"] for e in test_entries]
        assert len(ids) == len(set(ids)), "Duplicate entry_ids in test set"

    def test_no_overlap_between_calibration_and_test(self, calibration_entries, test_entries):
        cal_ids = {e["entry_id"] for e in calibration_entries}
        test_ids = {e["entry_id"] for e in test_entries}
        overlap = cal_ids & test_ids
        assert not overlap, f"entry_ids appear in both calibration and test: {overlap}"
