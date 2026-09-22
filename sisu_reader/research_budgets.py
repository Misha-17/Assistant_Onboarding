"""Runtime wall-clock allocations; the global research cutoff stays hard."""
from __future__ import annotations


def screening_cutoff(now: float, deadline: float, *, minimum_screen_s: float,
                     fraction: float = 0.20, ceiling_s: float = 15.0) -> float:
    """Give screening a bounded burst rather than the entire reader allowance.

    The 1.5x call minimum permits one viable screen and leaves scheduling
    headroom; it is an allocation, not a floating-point tolerance. If that
    allocation does not fit, normal minimum-window checks skip screening and
    retain manifests as unscreened candidates.
    """
    remaining = max(0.0, deadline - now)
    cap = max(1.5 * minimum_screen_s, ceiling_s)
    allowance = min(remaining, cap, max(1.5 * minimum_screen_s, remaining * fraction))
    return min(deadline, now + allowance)


def reader_window(now: float, *, global_cutoff: float, soft_cutoff: float,
                  minimum_reader_s: float, packets_read: int) -> float:
    """A soft wave allocation cannot prevent its first globally funded read."""
    remaining = global_cutoff - now
    if remaining < minimum_reader_s:
        return 0.0
    soft_remaining = soft_cutoff - now
    if packets_read and soft_remaining < minimum_reader_s:
        return 0.0
    return min(remaining, max(minimum_reader_s, soft_remaining))
