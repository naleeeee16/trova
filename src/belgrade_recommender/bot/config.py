"""Bot-wide tuneable constants — one place to change them all."""

from __future__ import annotations

# How long (seconds) to wait for group members to type after /find
COLLECTION_WINDOW_SECONDS: int = 300

# Results whose worst-member soft score is below this are noise, not shown
SCORE_FLOOR: float = 0.35

# If the best result's min_soft is below this the group has no usable consensus
NO_CONSENSUS_THRESHOLD: float = 0.35
