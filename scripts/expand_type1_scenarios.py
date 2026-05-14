"""
Expand Type 1 (instruction ambiguity) scenarios in nav_calibration.json and
nav_test.json to achieve 7 calibration + 3 test entries per sub-type.

Also back-fills the `ambiguity_subtype` field on all existing Type 1 entries.

Sub-type targets:
    ambiguous_target   cal 2→7 (+5)   test 3→3 (+0)
    ambiguous_action   cal 2→7 (+5)   test 0→3 (+3)
    missing_object     cal 5→7 (+2)   test 1→3 (+2)
    missing_action     cal 1→7 (+6)   test 1→3 (+2)
    missing_direction  cal 1→7 (+6)   test 1→3 (+2)
    missing_distance   cal 0→7 (+7)   test 1→3 (+2)
    no_uncertainty     cal 1→3 (+2)   test 2→3 (+1)   (needed for FPR eval)
"""

import json
from pathlib import Path

DATA_DIR = Path("baselines/introplan/data")
CAL_FILE = DATA_DIR / "nav_calibration.json"
TEST_FILE = DATA_DIR / "nav_test.json"

# ── Back-fill map for existing Type 1 entries ─────────────────────────────────

EXISTING_SUBTYPES = {
    "nav_003":  "ambiguous_target",
    "nav_007":  "ambiguous_action",
    "nav_010":  "missing_object",
    "nav_016":  "ambiguous_target",
    "nav_019":  "no_uncertainty",
    "nav_021":  "missing_object",
    "nav_022":  "missing_object",
    "nav_023":  "ambiguous_action",
    "nav_025":  "missing_object",
    "nav_026":  "missing_action",
    "nav_027":  "missing_object",
    "nav_028":  "missing_direction",
    "nav_t001": "ambiguous_target",
    "nav_t002": "ambiguous_target",
    "nav_t003": "missing_distance",
    "nav_t004": "missing_direction",
    "nav_t005": "missing_object",
    "nav_t025": "no_uncertainty",
    "nav_t026": "no_uncertainty",
    "nav_t031": "ambiguous_target",
    "nav_t032": "missing_action",
}

# ── New calibration entries ───────────────────────────────────────────────────

NEW_CAL = [
    # ── ambiguous_target (+5: nav_059–063) ────────────────────────────────────
    {
        "entry_id": "nav_059",
        "instruction": "Take me to the entrance",
        "terrain_description": "paved path approaching a large building complex with visible entrances on both the north and south sides",
        "uncertainty_type": 1,
        "ambiguity_subtype": "ambiguous_target",
        "options": {
            "A": "Navigate to the nearest entrance by default",
            "B": "Ask user: Which entrance do you mean — the north side or the south side?",
            "C": "Stop and wait in front of the building",
            "D": "Circle the building to identify the primary entrance",
        },
        "correct_option": "B",
        "reasoning": "Two entrances are visible and equidistant. Without more context the robot cannot determine which one the user means. A single clarifying question resolves the ambiguity with no movement wasted.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_060",
        "instruction": "Go to the meeting point",
        "terrain_description": "open park area with no designated meeting-point marker visible; several benches and a gazebo are distributed across the space",
        "uncertainty_type": 1,
        "ambiguity_subtype": "ambiguous_target",
        "options": {
            "A": "Navigate to the geometric center of the park as a default",
            "B": "Ask user: Where is the meeting point — near the gazebo or by the benches?",
            "C": "Stop and announce the robot cannot find the meeting point",
            "D": "Proceed to the largest open clearing in view",
        },
        "correct_option": "B",
        "reasoning": "No meeting point is marked in the robot's map. The term is specific but the location is unknown to the robot. A brief question identifies the correct destination immediately.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_061",
        "instruction": "Head to the usual spot",
        "terrain_description": "residential pathway at a familiar-seeming location; robot has no persistent memory of a previous 'usual spot' stored in session",
        "uncertainty_type": 1,
        "ambiguity_subtype": "ambiguous_target",
        "options": {
            "A": "Navigate to the last destination from this session",
            "B": "Ask user: Which spot do you mean — I don't have a saved location for 'the usual spot'",
            "C": "Stop and report that the location is unknown",
            "D": "Return to the session start point as a proxy",
        },
        "correct_option": "B",
        "reasoning": "The robot has no stored 'usual spot' for this user. Assuming the last session destination could be wrong. Asking takes one second and prevents a wasted trip.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_062",
        "instruction": "Navigate to the exit",
        "terrain_description": "enclosed park or campus with at least three visible gate exits; the forward path branches toward two of them",
        "uncertainty_type": 1,
        "ambiguity_subtype": "ambiguous_target",
        "options": {
            "A": "Take the nearest exit gate automatically",
            "B": "Ask user: Which exit — the main gate, the east gate, or the side gate?",
            "C": "Stop at the current location and announce multiple exits exist",
            "D": "Head to the exit that leads toward the most traffic",
        },
        "correct_option": "B",
        "reasoning": "Three exits are visible and the user's preferred one depends on their onward destination. Choosing the wrong exit could require significant backtracking.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_063",
        "instruction": "Take me back to where we started",
        "terrain_description": "mid-route on a winding park trail; the session began over 400 meters back and multiple path branches have been taken since",
        "uncertainty_type": 1,
        "ambiguity_subtype": "ambiguous_target",
        "options": {
            "A": "Retrace the last known path waypoints to the session origin",
            "B": "Ask user: Do you mean the park entrance where we began, or a different starting point?",
            "C": "Stop and report that the starting location is no longer in memory",
            "D": "Navigate to the nearest parking area as a likely starting point",
        },
        "correct_option": "B",
        "reasoning": "Session start is stored but the user may mean a specific landmark rather than the GPS start coordinate. A quick confirmation avoids taking the wrong route.",
        "source_image": None,
        "should_ask": True,
    },

    # ── ambiguous_action (+5: nav_064–068) ────────────────────────────────────
    {
        "entry_id": "nav_064",
        "instruction": "Handle the obstacle",
        "terrain_description": "a fallen branch partially blocking the path ahead; the robot can go around it on either side or stop before it",
        "uncertainty_type": 1,
        "ambiguity_subtype": "ambiguous_action",
        "options": {
            "A": "Autonomously reroute around the left side of the branch",
            "B": "Ask user: Should I go around the branch, wait for it to be moved, or stop here?",
            "C": "Stop immediately in front of the branch",
            "D": "Slow down and attempt to cross over the branch",
        },
        "correct_option": "B",
        "reasoning": "'Handle' is semantically underspecified. It could mean reroute, stop, request help, or attempt to cross. The preferred action depends on the user's urgency and accessibility needs.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_065",
        "instruction": "Deal with the puddle ahead",
        "terrain_description": "a wide puddle of unknown depth spanning roughly 70% of the path width; a narrow dry strip exists on the right edge",
        "uncertainty_type": 1,
        "ambiguity_subtype": "ambiguous_action",
        "options": {
            "A": "Navigate around the puddle using the dry strip on the right",
            "B": "Ask user: Should I go around the puddle on the right, cross through it, or find an alternate route?",
            "C": "Stop before the puddle and wait",
            "D": "Cross through the center of the puddle at reduced speed",
        },
        "correct_option": "B",
        "reasoning": "'Deal with' is ambiguous. Going around, crossing, or finding an alternate route all satisfy the instruction. The right action depends on whether the user needs speed, safety, or dryness.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_066",
        "instruction": "Manage the incline ahead",
        "terrain_description": "a moderate uphill grade beginning 10 meters ahead; the slope is passable but steeper than the previous path segment",
        "uncertainty_type": 1,
        "ambiguity_subtype": "ambiguous_action",
        "options": {
            "A": "Proceed up the incline at reduced speed",
            "B": "Ask user: Should I go up the slope slowly, find a flatter route around it, or stop here?",
            "C": "Stop before the slope and request manual guidance",
            "D": "Search the map for a level alternative route",
        },
        "correct_option": "B",
        "reasoning": "'Manage' the incline could mean slow down, reroute, or ask for manual assistance. For users with accessibility needs, the preferred action varies significantly.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_067",
        "instruction": "Take care of the wet section",
        "terrain_description": "a stretch of wet compressed leaves covering the path for approximately 3 meters; leaf surface is slippery and traversability is uncertain",
        "uncertainty_type": 1,
        "ambiguity_subtype": "ambiguous_action",
        "options": {
            "A": "Cross the wet section at minimum speed",
            "B": "Ask user: Should I cross the wet leaves carefully, go around through the grass, or wait for them to dry?",
            "C": "Stop and refuse to proceed until the surface is confirmed safe",
            "D": "Automatically divert onto the adjacent grass verge",
        },
        "correct_option": "B",
        "reasoning": "'Take care of' is vague. The robot has multiple valid responses (cross, divert, wait) and the user's risk tolerance and time constraints determine the right one.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_068",
        "instruction": "Work around the construction",
        "terrain_description": "a temporary construction barrier partially blocking the path; a detour via an adjacent grass area is visible on the left; the construction may or may not affect the target destination",
        "uncertainty_type": 1,
        "ambiguity_subtype": "ambiguous_action",
        "options": {
            "A": "Take the visible grass detour on the left",
            "B": "Ask user: Should I use the grass detour on the left, find another route, or wait for the construction to clear?",
            "C": "Stop and report the path blocked",
            "D": "Continue straight and attempt to pass through a narrow gap in the barrier",
        },
        "correct_option": "B",
        "reasoning": "'Work around' could mean the specific visible detour or a different route the user prefers. Choosing the wrong detour may take the user far from their destination.",
        "source_image": None,
        "should_ask": True,
    },

    # ── missing_object (+2: nav_069–070) ──────────────────────────────────────
    {
        "entry_id": "nav_069",
        "instruction": "Find me somewhere accessible",
        "terrain_description": "urban park with multiple paths; a paved ramp, a gravel shortcut, and a stepped path are all visible options ahead",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_object",
        "options": {
            "A": "Navigate to the nearest paved ramp as the most accessible option",
            "B": "Ask user: What kind of accessible spot are you looking for — a ramp, a flat seating area, or a smooth path?",
            "C": "Stop and search the map for all wheelchair-accessible locations",
            "D": "Follow the widest path as a proxy for accessibility",
        },
        "correct_option": "B",
        "reasoning": "'Accessible' could refer to physical accessibility features, proximity, or open space. Without knowing what the user needs, any autonomous choice risks selecting the wrong type of location.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_070",
        "instruction": "Take me somewhere with better lighting",
        "terrain_description": "evening conditions; current path is poorly lit; a well-lit parking area is 50 meters ahead, a lit covered shelter is 30 meters to the right",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_object",
        "options": {
            "A": "Navigate to the nearest well-lit area (the shelter, 30m)",
            "B": "Ask user: Would you prefer the covered shelter nearby or the parking area further ahead?",
            "C": "Remain stationary and activate the robot's lights",
            "D": "Head toward the brightest visible light source",
        },
        "correct_option": "B",
        "reasoning": "Two lit options exist with different distances and environments. The shelter is closer but the parking area may be more convenient. The user's preference depends on where they are going next.",
        "source_image": None,
        "should_ask": True,
    },

    # ── missing_action (+6: nav_071–076) ──────────────────────────────────────
    {
        "entry_id": "nav_071",
        "instruction": "The fountain",
        "terrain_description": "large open park; a decorative fountain is visible 80 meters ahead to the right of the main path",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_action",
        "options": {
            "A": "Navigate to the fountain as the likely intended destination",
            "B": "Ask user: What would you like me to do — navigate to the fountain, stop near it, or something else?",
            "C": "Stop and wait for a complete instruction",
            "D": "Face the fountain and pause for user input",
        },
        "correct_option": "B",
        "reasoning": "The instruction is a noun with no action verb. The user may want to go to the fountain, stop beside it, turn toward it, or use it as a waypoint. Guessing the intended action risks misnavigating.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_072",
        "instruction": "That puddle",
        "terrain_description": "a wide muddy puddle spanning the path center; dry ground exists on both sides",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_action",
        "options": {
            "A": "Automatically route around the puddle",
            "B": "Ask user: What should I do about the puddle — go around it, or are you pointing it out for another reason?",
            "C": "Stop before the puddle and wait",
            "D": "Cross the puddle at minimum speed",
        },
        "correct_option": "B",
        "reasoning": "The user may be pointing out the puddle as a hazard to avoid, or they may want the robot to stop, or they may be testing whether the robot sees it. No action verb means the intent is unknown.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_073",
        "instruction": "The crosswalk ahead",
        "terrain_description": "a marked pedestrian crosswalk 15 meters ahead; pedestrian signal is currently red",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_action",
        "options": {
            "A": "Stop at the crosswalk and wait for a green signal",
            "B": "Ask user: Should I stop at the crosswalk, cross when the signal turns green, or take a different route?",
            "C": "Halt immediately and request manual guidance",
            "D": "Cross the road at the crosswalk regardless of signal",
        },
        "correct_option": "B",
        "reasoning": "Without an action verb, 'the crosswalk ahead' could be a directive to stop, proceed, or avoid. The robot should not cross against a red signal without confirmation, and should not stop unnecessarily if the user wants to cross.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_074",
        "instruction": "Your left side",
        "terrain_description": "wide straight path; a seating area is visible on the left edge, as well as a side branch path leading left",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_action",
        "options": {
            "A": "Move toward the left side of the path",
            "B": "Ask user: What should I do on the left — move over, turn left, or go to the seating area?",
            "C": "Stop and face left",
            "D": "Take the left branch path",
        },
        "correct_option": "B",
        "reasoning": "Referencing the left side without an action verb is ambiguous between lateral repositioning, turning, or navigating to a specific left-side landmark.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_075",
        "instruction": "That steep section",
        "terrain_description": "a steep downhill section begins 20 meters ahead; the path narrows and handrails are absent; an alternate flatter route exists to the right",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_action",
        "options": {
            "A": "Proceed down the steep section at reduced speed",
            "B": "Ask user: What should I do about the steep section — go down it, take the flatter route, or stop?",
            "C": "Stop before the steep section and wait",
            "D": "Automatically take the flatter alternate route",
        },
        "correct_option": "B",
        "reasoning": "Pointing out a terrain feature without a verb leaves the robot's action undefined. The correct response depends on whether the user wants to descend, reroute, or is warning the robot for awareness.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_076",
        "instruction": "The gravel section",
        "terrain_description": "the paved path transitions to loose gravel for approximately 20 meters; a paved bypass exists along the left edge",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_action",
        "options": {
            "A": "Cross the gravel section at reduced speed",
            "B": "Ask user: Should I cross the gravel, take the paved bypass on the left, or slow down and check with you first?",
            "C": "Stop before the gravel and refuse to proceed",
            "D": "Automatically route onto the paved bypass",
        },
        "correct_option": "B",
        "reasoning": "Naming a terrain feature without specifying an action leaves the robot's response undefined. Gravel may be uncomfortable or risky for the user, or it may be perfectly acceptable.",
        "source_image": None,
        "should_ask": True,
    },

    # ── missing_direction (+6: nav_077–082) ───────────────────────────────────
    {
        "entry_id": "nav_077",
        "instruction": "Turn at the crosswalk",
        "terrain_description": "approaching a crosswalk intersection with left-turn and right-turn lanes; both directions lead to valid paths",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_direction",
        "options": {
            "A": "Turn right at the crosswalk (most common default)",
            "B": "Ask user: Should I turn left or right at the crosswalk?",
            "C": "Stop at the crosswalk and wait",
            "D": "Continue straight through the crosswalk",
        },
        "correct_option": "B",
        "reasoning": "The instruction specifies where to turn but not which direction. Turning the wrong way could take the user significantly off course.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_078",
        "instruction": "Go the other way",
        "terrain_description": "currently heading north on a bidirectional path; 'the other way' could mean south (reverse), or east/west at the upcoming branch",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_direction",
        "options": {
            "A": "Reverse direction and head south",
            "B": "Ask user: Do you mean turn around and go back, or take a different branch at the next fork?",
            "C": "Stop and wait for clarification",
            "D": "Turn 90 degrees and proceed on the cross path",
        },
        "correct_option": "B",
        "reasoning": "'The other way' is relative and ambiguous at a branching path. It could mean reverse course or take a side branch; the wrong interpretation could require significant backtracking.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_079",
        "instruction": "Bear to the side",
        "terrain_description": "wide straight path with a pedestrian group ahead; space exists on both the left and right sides to pass them",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_direction",
        "options": {
            "A": "Bear left as the wider gap appears to be on the left",
            "B": "Ask user: Should I bear left or right to pass the group?",
            "C": "Stop and wait for the group to move",
            "D": "Announce approach and let the pedestrians step aside",
        },
        "correct_option": "B",
        "reasoning": "'Bear to the side' specifies a lateral adjustment but not which side. Both sides have passable space; guessing incorrectly may cause a collision with the other pedestrians.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_080",
        "instruction": "Veer off the path here",
        "terrain_description": "paved main path with grass on both sides; a shortcut through the grass leads to two different landmarks on left and right",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_direction",
        "options": {
            "A": "Veer left off the path toward the nearer landmark",
            "B": "Ask user: Should I veer left toward the shelter or right toward the parking area?",
            "C": "Stop on the current path and wait",
            "D": "Move slightly off-path to the right as a default",
        },
        "correct_option": "B",
        "reasoning": "Leaving the main path in the wrong direction could take the user to an unintended destination. The instruction specifies the action but not the direction.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_081",
        "instruction": "Take the alternate route",
        "terrain_description": "a main paved path and two alternate routes are visible — one through a gravel path on the left, one via a longer paved detour on the right",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_direction",
        "options": {
            "A": "Take the shorter gravel alternate on the left",
            "B": "Ask user: Which alternate route — the shorter gravel path on the left or the longer paved route on the right?",
            "C": "Continue on the main path",
            "D": "Take the longest visible alternate as the most distinct alternative",
        },
        "correct_option": "B",
        "reasoning": "Two alternates exist with different surfaces and distances. 'Alternate route' does not identify which one, and the wrong choice may be unsuitable for the user's mobility needs.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_082",
        "instruction": "Go around the bench",
        "terrain_description": "a bench positioned on the right side of the path; enough space to pass on either the left (path side) or the right (grass side)",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_direction",
        "options": {
            "A": "Pass on the left of the bench (path side)",
            "B": "Ask user: Should I go around the bench on the left side or the right side?",
            "C": "Stop before the bench and wait",
            "D": "Go around on the right (grass side) as it avoids other pedestrians",
        },
        "correct_option": "B",
        "reasoning": "The instruction specifies the maneuver but not the direction. Left means staying on the smooth path; right means going onto grass. The correct side depends on the user's mobility needs and comfort.",
        "source_image": None,
        "should_ask": True,
    },

    # ── missing_distance (+7: nav_083–089) ────────────────────────────────────
    {
        "entry_id": "nav_083",
        "instruction": "Move a bit to the right",
        "terrain_description": "wide path with a pedestrian approaching from the left; enough space to shift laterally by 20 cm, 50 cm, or 1 meter",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_distance",
        "options": {
            "A": "Shift 30 cm to the right as a reasonable default",
            "B": "Ask user: How far right — just a small shift or move to the right edge of the path?",
            "C": "Move to the far right edge of the path",
            "D": "Stop in place and yield to the approaching pedestrian",
        },
        "correct_option": "B",
        "reasoning": "'A bit' is subjective — a small adjustment vs. moving fully to the right edge makes a significant difference for passing pedestrians or repositioning near a destination.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_084",
        "instruction": "Back up slightly",
        "terrain_description": "robot is stopped near the edge of a paved area; a curb drop is 40 cm behind; 'slightly' could mean 10 cm or 50 cm",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_distance",
        "options": {
            "A": "Reverse 20 cm as a conservative 'slight' movement",
            "B": "Ask user: How far should I back up — a few centimeters or further back from the curb?",
            "C": "Refuse to reverse due to nearby curb drop",
            "D": "Reverse until the user says stop",
        },
        "correct_option": "B",
        "reasoning": "'Slightly' near a curb drop is safety-critical. The robot needs to know whether the user means a very small adjustment or a larger repositioning to safely execute the command.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_085",
        "instruction": "Go ahead a little",
        "terrain_description": "stopped at a path junction; 'a little' could mean 1 meter (to cross the junction) or 5 meters (to the next landmark)",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_distance",
        "options": {
            "A": "Move forward 2 meters as a default interpretation",
            "B": "Ask user: How far ahead — just past the junction or all the way to the bench?",
            "C": "Move forward until an obstacle is detected",
            "D": "Proceed to the next named waypoint on the map",
        },
        "correct_option": "B",
        "reasoning": "At a junction, moving 1 meter vs. 5 meters leads to completely different positions. 'A little' is too vague to act on safely.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_086",
        "instruction": "Come closer",
        "terrain_description": "robot is 3 meters from the user who is seated on a bench; 'closer' could mean 1 meter away or directly adjacent",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_distance",
        "options": {
            "A": "Move to 1 meter from the user as a comfortable default",
            "B": "Ask user: How close — right next to you or just a meter closer?",
            "C": "Move forward until the user says stop",
            "D": "Move until the proximity sensor detects a person",
        },
        "correct_option": "B",
        "reasoning": "Stopping distance from a seated user depends on whether they need to use the robot's controls, pick up a bag from it, or simply want it nearby. A single question prevents an uncomfortable or unusable position.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_087",
        "instruction": "Move further along the path",
        "terrain_description": "on a long straight path with multiple benches and landmarks spaced 10–50 meters apart; 'further along' provides no stopping criterion",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_distance",
        "options": {
            "A": "Move 10 meters forward and stop",
            "B": "Ask user: How far along — to the next bench, the shelter, or a specific distance?",
            "C": "Continue until an obstacle or path end is reached",
            "D": "Move to the midpoint of the current path segment",
        },
        "correct_option": "B",
        "reasoning": "Without a stopping criterion, 'further along' has no well-defined terminal point. Stopping too early or too late both fail to satisfy the user's intent.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_088",
        "instruction": "Just a little bit further",
        "terrain_description": "approaching a road crossing; 'a little bit' could mean stopping before the crossing or moving past it onto the far sidewalk",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_distance",
        "options": {
            "A": "Move 1 meter forward and stop",
            "B": "Ask user: Do you mean stop before the road crossing or continue past it to the far side?",
            "C": "Continue until the robot reaches the road edge",
            "D": "Cross the road and stop on the far sidewalk",
        },
        "correct_option": "B",
        "reasoning": "Near a road crossing, even a small ambiguity in distance is safety-critical. The robot must confirm whether it should stop before or after the crossing.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_089",
        "instruction": "Stop somewhere around here",
        "terrain_description": "current location is in the middle of an open plaza; benches, a shelter, and open paved areas are all within 15 meters",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_distance",
        "options": {
            "A": "Stop at the current position immediately",
            "B": "Ask user: Should I stop right here, or move to the bench or the shelter nearby?",
            "C": "Navigate to the nearest seating area",
            "D": "Stop at the geometrically central point of the plaza",
        },
        "correct_option": "B",
        "reasoning": "'Around here' and 'somewhere' are both vague. The user may want a specific nearby feature (bench, shelter) or just any stopping point. A brief question allows a useful placement.",
        "source_image": None,
        "should_ask": True,
    },

    # ── no_uncertainty (+2: nav_090–091) ──────────────────────────────────────
    {
        "entry_id": "nav_090",
        "instruction": "Turn left at the next crosswalk",
        "terrain_description": "straight path approaching a marked pedestrian crosswalk with clear left-turn lane; no obstructions",
        "uncertainty_type": 1,
        "ambiguity_subtype": "no_uncertainty",
        "options": {
            "A": "Turn left at the crosswalk as instructed",
            "B": "Ask user: Do you mean turn left at the crosswalk ahead?",
            "C": "Stop at the crosswalk and wait",
            "D": "Continue straight past the crosswalk",
        },
        "correct_option": "A",
        "reasoning": "The instruction specifies both action (turn) and direction (left) at a named landmark (crosswalk). All required information is present; asking would be unnecessary and irritating.",
        "source_image": None,
        "should_ask": False,
    },
    {
        "entry_id": "nav_091",
        "instruction": "Go straight for 20 meters",
        "terrain_description": "clear straight paved path; no obstructions; path continues well beyond 20 meters",
        "uncertainty_type": 1,
        "ambiguity_subtype": "no_uncertainty",
        "options": {
            "A": "Move straight ahead for 20 meters and stop",
            "B": "Ask user: Do you mean exactly 20 meters or approximately?",
            "C": "Move forward until an obstacle is detected",
            "D": "Move forward to the next intersection",
        },
        "correct_option": "A",
        "reasoning": "Action (go straight), distance (20 meters), and direction (forward) are all specified. The instruction is unambiguous and can be executed directly.",
        "source_image": None,
        "should_ask": False,
    },
]

# ── New test entries ──────────────────────────────────────────────────────────

NEW_TEST = [
    # ── ambiguous_action (+3: nav_t033–035) ───────────────────────────────────
    {
        "entry_id": "nav_t033",
        "instruction": "Handle the hill",
        "terrain_description": "a moderate uphill grade begins 8 meters ahead; the main path goes up it, but a flatter bypass exists on the right",
        "uncertainty_type": 1,
        "ambiguity_subtype": "ambiguous_action",
        "options": {
            "A": "Ascend the hill at reduced speed",
            "B": "Ask user: Should I go up the hill, take the flat bypass, or stop here?",
            "C": "Stop before the hill",
            "D": "Automatically take the flatter bypass route",
        },
        "correct_option": "B",
        "reasoning": "'Handle' is semantically vague — ascending, rerouting, and stopping are all valid interpretations. The appropriate action depends on user needs and destination.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_t034",
        "instruction": "Deal with it",
        "terrain_description": "a wet muddy patch covers the direct forward path; a narrow dry path exists around it to the left",
        "uncertainty_type": 1,
        "ambiguity_subtype": "ambiguous_action",
        "options": {
            "A": "Route around the mud patch on the left",
            "B": "Ask user: What would you like me to do — go around the mud, cross it slowly, or stop?",
            "C": "Stop before the mud and wait",
            "D": "Cross through the mud at minimum speed",
        },
        "correct_option": "B",
        "reasoning": "'Deal with it' provides no specific action. The robot cannot determine whether the user wants avoidance, slow crossing, or stopping without asking.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_t035",
        "instruction": "Work around the construction zone",
        "terrain_description": "a construction barrier closes the direct path; two detour options exist — a shorter unpaved route left and a longer paved route further right",
        "uncertainty_type": 1,
        "ambiguity_subtype": "ambiguous_action",
        "options": {
            "A": "Take the shorter unpaved detour on the left",
            "B": "Ask user: Should I take the shorter dirt detour or the longer paved route around the construction?",
            "C": "Stop and report the path blocked",
            "D": "Wait for the construction to clear",
        },
        "correct_option": "B",
        "reasoning": "'Work around' specifies that a detour is needed but not which one. Surface type and distance differ significantly between options.",
        "source_image": None,
        "should_ask": True,
    },

    # ── missing_object (+2: nav_t036–037) ─────────────────────────────────────
    {
        "entry_id": "nav_t036",
        "instruction": "Take me somewhere accessible",
        "terrain_description": "approaching a park with a ramp entrance on the left and stepped entrance on the right; a flat open seating area is visible beyond both entrances",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_object",
        "options": {
            "A": "Use the ramp entrance as the most accessible route",
            "B": "Ask user: Are you looking for an accessible entrance, a flat seating area, or somewhere specific in the park?",
            "C": "Stop at the park boundary and wait",
            "D": "Navigate to the geometric center of the park",
        },
        "correct_option": "B",
        "reasoning": "Multiple accessible features exist. Without knowing whether the user needs accessible entry, rest area, or a different accessible feature, any autonomous choice may be wrong.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_t037",
        "instruction": "Find me a shaded area",
        "terrain_description": "open sunny park; a large tree canopy provides shade 40 meters to the right, a covered shelter is 60 meters straight ahead",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_object",
        "options": {
            "A": "Navigate to the nearer shaded area under the trees (40 m)",
            "B": "Ask user: Would you prefer the shaded tree area to the right or the covered shelter straight ahead?",
            "C": "Stop and scan for all shaded areas",
            "D": "Navigate to the shelter as the most reliably shaded option",
        },
        "correct_option": "B",
        "reasoning": "Two shaded options exist with different environments. The nearer is natural shade (may not be wheelchair-accessible); the farther is a covered structure (more reliable shade). User preference determines the right choice.",
        "source_image": None,
        "should_ask": True,
    },

    # ── missing_action (+2: nav_t038–039) ─────────────────────────────────────
    {
        "entry_id": "nav_t038",
        "instruction": "That puddle",
        "terrain_description": "a large puddle of uncertain depth occupying 60% of the path; a narrow dry strip on the left allows passing",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_action",
        "options": {
            "A": "Route around the puddle using the dry strip",
            "B": "Ask user: What should I do about the puddle — go around it, cross it, or stop here?",
            "C": "Stop before the puddle and wait",
            "D": "Cross through the puddle at minimum speed",
        },
        "correct_option": "B",
        "reasoning": "Pointing to a puddle without a verb is ambiguous. The user may want avoidance, slow crossing, a stop, or may simply be flagging it as a hazard to note.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_t039",
        "instruction": "The shelter ahead",
        "terrain_description": "a covered bus shelter is visible 25 meters ahead on the right side of the path; the path continues past it",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_action",
        "options": {
            "A": "Navigate to the shelter and stop beside it",
            "B": "Ask user: Should I take you to the shelter, stop in front of it, or continue past it?",
            "C": "Stop immediately and face the shelter",
            "D": "Continue past the shelter to the next destination",
        },
        "correct_option": "B",
        "reasoning": "Naming a landmark without a verb leaves the robot's action undefined. The user may want to go to the shelter, rest there, meet someone, or is simply pointing it out.",
        "source_image": None,
        "should_ask": True,
    },

    # ── missing_direction (+2: nav_t040–041) ──────────────────────────────────
    {
        "entry_id": "nav_t040",
        "instruction": "Turn at the next intersection",
        "terrain_description": "approaching a 4-way intersection 20 meters ahead; all four directions lead to valid paths; no prior destination context exists",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_direction",
        "options": {
            "A": "Turn right as the default at an unmarked intersection",
            "B": "Ask user: Should I turn left or right at the intersection ahead?",
            "C": "Stop at the intersection and wait",
            "D": "Continue straight through the intersection",
        },
        "correct_option": "B",
        "reasoning": "The instruction specifies when to turn but not which direction. At a 4-way intersection, turning left vs. right leads to completely different destinations.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_t041",
        "instruction": "Bear to the side when you get there",
        "terrain_description": "a narrowing path ahead caused by parked bicycles; space exists on both left (grass) and right (fence) to navigate past",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_direction",
        "options": {
            "A": "Bear left toward the grass as the wider gap",
            "B": "Ask user: Should I bear left onto the grass or right along the fence?",
            "C": "Stop before the narrow section and wait",
            "D": "Announce the narrowing and proceed straight",
        },
        "correct_option": "B",
        "reasoning": "'To the side' is directionally unspecified. Left goes over grass (potentially unsuitable); right goes along a fence (potentially too narrow). The correct side depends on the user's mobility needs.",
        "source_image": None,
        "should_ask": True,
    },

    # ── missing_distance (+2: nav_t042–043) ───────────────────────────────────
    {
        "entry_id": "nav_t042",
        "instruction": "Come a little closer",
        "terrain_description": "robot is positioned 4 meters from a seated user at a picnic table; 'closer' could mean 1 meter away or right beside the table",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_distance",
        "options": {
            "A": "Move to 1.5 meters from the user",
            "B": "Ask user: How close — within arm's reach or just a bit nearer?",
            "C": "Move forward until the proximity sensor triggers",
            "D": "Move to 0.5 meters (arm's reach) from the table",
        },
        "correct_option": "B",
        "reasoning": "'A little closer' is relative and context-dependent. Whether the user needs to reach the controls, place a bag, or just wants company determines the appropriate stopping distance.",
        "source_image": None,
        "should_ask": True,
    },
    {
        "entry_id": "nav_t043",
        "instruction": "Back up some",
        "terrain_description": "robot is stopped at a narrow gate; backing up 20 cm clears the gate; backing up 1 meter clears the entire entrance area",
        "uncertainty_type": 1,
        "ambiguity_subtype": "missing_distance",
        "options": {
            "A": "Reverse 30 cm as a minimal 'some' movement",
            "B": "Ask user: How far back — just past the gate or further back to clear the whole entrance?",
            "C": "Reverse until an obstacle behind is detected",
            "D": "Reverse 1 meter to fully clear the entrance",
        },
        "correct_option": "B",
        "reasoning": "The functional difference between backing up minimally vs. fully depends on whether the user needs to reposition themselves or allow someone else through the gate. 'Some' is too vague to act on safely.",
        "source_image": None,
        "should_ask": True,
    },

    # ── no_uncertainty (+1: nav_t044) ─────────────────────────────────────────
    {
        "entry_id": "nav_t044",
        "instruction": "Stop at the bench 10 meters ahead on the right",
        "terrain_description": "clear straight path; a bench is visible on the right edge approximately 10 meters ahead; no obstructions",
        "uncertainty_type": 1,
        "ambiguity_subtype": "no_uncertainty",
        "options": {
            "A": "Navigate to the bench 10 meters ahead on the right and stop",
            "B": "Ask user: Is that the bench you mean?",
            "C": "Stop at the current position",
            "D": "Stop at the nearest bench regardless of distance",
        },
        "correct_option": "A",
        "reasoning": "The instruction specifies the action (stop), target (bench), distance (10 meters), and side (right). All required information is present; no clarification is needed.",
        "source_image": None,
        "should_ask": False,
    },
]


def main() -> None:
    # Load files
    with open(CAL_FILE) as f:
        cal = json.load(f)
    with open(TEST_FILE) as f:
        test = json.load(f)

    # Back-fill ambiguity_subtype on existing Type 1 entries
    for entry in cal + test:
        if entry["uncertainty_type"] == 1 and "ambiguity_subtype" not in entry:
            entry["ambiguity_subtype"] = EXISTING_SUBTYPES.get(entry["entry_id"], "unknown")

    # Append new entries
    cal.extend(NEW_CAL)
    test.extend(NEW_TEST)

    # Write back
    with open(CAL_FILE, "w") as f:
        json.dump(cal, f, indent=2)
    with open(TEST_FILE, "w") as f:
        json.dump(test, f, indent=2)

    # Summary
    from collections import Counter
    t1_cal = [e for e in cal if e["uncertainty_type"] == 1]
    t1_test = [e for e in test if e["uncertainty_type"] == 1]
    cal_sub = Counter(e.get("ambiguity_subtype", "?") for e in t1_cal)
    test_sub = Counter(e.get("ambiguity_subtype", "?") for e in t1_test)

    print(f"Calibration: {len(cal)} total, {len(t1_cal)} Type 1")
    print(f"  Sub-type distribution: {dict(cal_sub)}")
    print(f"Test: {len(test)} total, {len(t1_test)} Type 1")
    print(f"  Sub-type distribution: {dict(test_sub)}")


if __name__ == "__main__":
    main()
