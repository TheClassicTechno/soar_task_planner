"""
User profile abstraction for personalized question generation.

A UserProfile encodes three dimensions of communication preference:
  verbosity        — how much detail the robot should include
  expertise        — how technical the language can be
  preferred_format — whether to ask a question, make a statement, or offer options

Profiles are stored in a UserProfileStore and retrieved by user_id.
An unknown user_id returns the DEFAULT_PROFILE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class UserProfile:
    """
    Immutable description of a user's communication preferences.

    Attributes:
        user_id:          Unique string identifier for this user.
        verbosity:        "terse" | "standard" | "verbose"
        expertise:        "novice" | "intermediate" | "expert"
        preferred_format: "question" | "statement" | "option_list"
        name:             Optional display name for logging.
    """

    user_id: str
    verbosity: str = "standard"
    expertise: str = "intermediate"
    preferred_format: str = "question"
    name: Optional[str] = None

    def __post_init__(self) -> None:
        valid_verbosity = {"terse", "standard", "verbose"}
        valid_expertise = {"novice", "intermediate", "expert"}
        valid_format = {"question", "statement", "option_list"}
        if self.verbosity not in valid_verbosity:
            raise ValueError(f"verbosity must be one of {valid_verbosity}")
        if self.expertise not in valid_expertise:
            raise ValueError(f"expertise must be one of {valid_expertise}")
        if self.preferred_format not in valid_format:
            raise ValueError(f"preferred_format must be one of {valid_format}")


DEFAULT_PROFILE = UserProfile(
    user_id="default",
    verbosity="standard",
    expertise="intermediate",
    preferred_format="question",
    name="Default User",
)


class UserProfileStore:
    """
    In-memory store mapping user_id → UserProfile.

    Unknown user IDs return DEFAULT_PROFILE rather than raising an error.

    Usage:
        store = UserProfileStore()
        store.register(UserProfile("alice", verbosity="terse", expertise="expert"))
        profile = store.get("alice")  # → terse expert profile
        profile = store.get("unknown_id")  # → DEFAULT_PROFILE
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, UserProfile] = {}

    def register(self, profile: UserProfile) -> None:
        """Add or replace a user profile."""
        self._profiles[profile.user_id] = profile

    def get(self, user_id: str) -> UserProfile:
        """
        Return the profile for user_id, or DEFAULT_PROFILE if not found.

        Args:
            user_id: The user's unique identifier.

        Returns:
            Registered UserProfile or DEFAULT_PROFILE.
        """
        return self._profiles.get(user_id, DEFAULT_PROFILE)

    def remove(self, user_id: str) -> None:
        """Remove a user profile if it exists."""
        self._profiles.pop(user_id, None)

    def __len__(self) -> int:
        return len(self._profiles)


def describe_profile_for_prompt(profile: UserProfile) -> str:
    """
    Build a concise natural-language description of the profile for LLM prompts.

    Returns a multi-line string that can be injected into the question generation
    prompt so the LLM understands the user's expected communication style.

    Args:
        profile: UserProfile to describe.

    Returns:
        Multi-line string suitable for inclusion in an LLM prompt.
    """
    verbosity_map = {
        "terse": "one sentence maximum, no explanation",
        "standard": "two sentences with brief context",
        "verbose": "full explanation including sensor diagnostics and alternatives",
    }
    expertise_map = {
        "novice": "avoid technical jargon, use plain language and give explicit choices",
        "intermediate": "standard robotics navigation phrasing",
        "expert": "include traversability score, sensor confidence, and trajectory geometry",
    }
    format_map = {
        "question": "ask a direct yes/no or choice question",
        "statement": "make a declarative statement with an implied request",
        "option_list": "present numbered options for the user to choose from",
    }
    lines = [
        "User communication profile:",
        f"  - Verbosity: {verbosity_map[profile.verbosity]}",
        f"  - Expertise: {expertise_map[profile.expertise]}",
        f"  - Format:    {format_map[profile.preferred_format]}",
    ]
    if profile.name:
        lines.insert(1, f"  - Name: {profile.name}")
    return "\n".join(lines)
