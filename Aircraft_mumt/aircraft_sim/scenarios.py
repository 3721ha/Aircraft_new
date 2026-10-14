import random
from typing import Optional

from .config import SimConfig


def boundary_scenario(n: int = 3) -> dict:
    positions = [(0.0, 0.0, 5000.0), (700.0, 0.0, 5000.0)] + [(3000.0 + i * 1200.0, 2000.0, 5200.0) for i in range(max(0, n - 2))]
    return {"positions": positions[:n], "resources": [0.35] + [0.8] * max(0, n - 1), "id_confidence": {0: 0.45}}


def conflict_scenario(n: int = 3) -> dict:
    return {"positions": [(-1000.0, 0.0, 5000.0), (1000.0, 0.0, 5000.0), (0.0, 2200.0, 5000.0)][:n], "resources": ([0.24, 0.8, 0.8] + [0.8] * max(0, n - 3))[:n], "c2_lost": {0: True}, "id_confidence": {0: 0.5}}


def protocol_conflict_scenario(n: int = 3) -> dict:
    """Exercise authorization, stale information, handover and support rules."""
    return {
        "positions": [(0.0, 0.0, 5000.0), (1800.0, 250.0, 5000.0), (3600.0, -250.0, 5000.0)][:n],
        "resources": ([0.16, 0.72, 0.86] + [0.8] * max(0, n - 3))[:n],
        "authorization": {0: False},
        "data_age": {0: 7},
        "intent_unsent": {1: True},
        "source_disagreement": {0: 0.65},
        "support_requested": {0: True},
        "task_progress": {0: 0.05},
    }


def support_separation_conflict_scenario(n: int = 3) -> dict:
    """Support deadline competes with an approaching helper's safe trend.

    Aircraft 0 and 1 approach head-on with enough initial distance for a
    residual turn to recover.  Aircraft 1 has the largest resource estimate,
    so the nominal team choice is the useful but geometrically constrained
    helper rather than the distant fallback aircraft.
    """
    return {
        # 2000 m remains above the 800 m hard separation margin while the
        # 4.5 s closing-time forecast is visible early enough for the
        # bounded safe-trend response window.
        "positions": [(0.0, 0.0, 5000.0), (2000.0, 0.0, 5000.0), (0.0, 3000.0, 5100.0)][:n],
        "resources": ([0.45, 0.90, 0.70] + [0.8] * max(0, n - 3))[:n],
        "support_requested": {0: True},
        "task_progress": {i: 0.10 for i in range(n)},
    }


def coverage_resource_conflict_scenario(n: int = 3, initially_feasible: bool = False) -> dict:
    """The nominal cover member must exit and another member can take over.

    The legacy default starts below the resource threshold for unit tests and
    immediate rule-trigger checks.  Formal conflict experiments request an
    initially feasible variant, where the cover member crosses the threshold
    only after sustained resource drain.
    """
    return {
        "positions": [(0.0, 0.0, 5000.0), (0.0, 1800.0, 5100.0), (0.0, 3600.0, 5200.0)][:n],
        "resources": ([(0.24 if initially_feasible else 0.12), 0.72, 0.90] + [0.8] * max(0, n - 3))[:n],
        "task_progress": {i: 0.10 for i in range(n)},
    }


def deadline_information_conflict_scenario(n: int = 3) -> dict:
    """A critical task is urgent while identification information is stale."""
    return {
        "positions": [(0.0, 0.0, 5000.0), (0.0, 1900.0, 5100.0), (0.0, 3800.0, 5200.0)][:n],
        "resources": [0.85] * n,
        "id_confidence": {0: 0.38},
        "data_age": {0: 8},
        "task_progress": {0: 0.05},
    }


def dual_support_conflict_scenario(n: int = 3) -> dict:
    """Two members request support while only one capable helper remains."""
    return {
        "positions": [(0.0, 0.0, 5000.0), (0.0, 1700.0, 5050.0), (0.0, 3400.0, 5100.0)][:n],
        "resources": ([0.45, 0.43, 0.82] + [0.8] * max(0, n - 3))[:n],
        "support_requested": {0: True, 1: True},
        "task_progress": {i: 0.10 for i in range(n)},
    }


def mission_recovery_conflict_scenario(n: int = 3, initially_feasible: bool = False) -> dict:
    """The task-active lead must recover while teammates retain capability."""
    return {
        "positions": [(0.0, 0.0, 600.0), (0.0, 2000.0, 5100.0), (0.0, 4000.0, 5200.0)][:n],
        "resources": ([(0.28 if initially_feasible else 0.18), 0.78, 0.86] + [0.8] * max(0, n - 3))[:n],
        "energies": ([(0.28 if initially_feasible else 0.20), 0.82, 0.88] + [0.8] * max(0, n - 3))[:n],
        "task_progress": {0: 0.35},
    }


def communication_intent_conflict_scenario(n: int = 3) -> dict:
    """Lost-link fallback competes with the requirement to share intent."""
    return {
        "positions": [(0.0, 0.0, 5000.0), (0.0, 2200.0, 5100.0), (0.0, 4400.0, 5200.0)][:n],
        "resources": ([0.82, 0.82, 0.82] + [0.8] * max(0, n - 3))[:n],
        "c2_lost": {0: True},
        "intent_unsent": {0: True},
        "task_authority": {0: True, 1: False, 2: False},
    }


def capability_dwell_conflict_scenario(n: int = 3, initially_feasible: bool = False) -> dict:
    """Capability loss requires an emergency role change before dwell ends.

    The feasible variant starts just above the capability-loss threshold and
    relies on resource depletion to trigger the fault during the rollout.
    """
    return {
        "positions": [(0.0, 0.0, 5000.0), (0.0, 2200.0, 5100.0), (0.0, 4400.0, 5200.0)][:n],
        "resources": ([(0.28 if initially_feasible else 0.82), 0.85, 0.85] + [0.8] * max(0, n - 3))[:n],
        "capabilities": ([(0.55 if initially_feasible else 0.30), 1.0, 1.0] + [0.8] * max(0, n - 3))[:n],
        "task_progress": {0: 0.25, 1: 0.25, 2: 0.25},
    }


def risk_continuity_conflict_scenario(n: int = 3) -> dict:
    """One member loses risk budget while the disconnected lead retains intent."""
    return {
        "positions": [(0.0, 0.0, 5000.0), (0.0, 2200.0, 5100.0), (0.0, 4400.0, 5200.0)][:n],
        # Keep the member above the F04 low-resource threshold while its
        # combined budget remains below the scenario-specific F06 threshold.
        "resources": ([0.82, 0.36, 0.82] + [0.8] * max(0, n - 3))[:n],
        "energies": ([0.82, 0.22, 0.82] + [0.8] * max(0, n - 3))[:n],
        "c2_lost": {0: True},
        "task_authority": {0: True, 1: False, 2: False},
        "task_progress": {0: 0.35, 1: 0.35, 2: 0.35},
    }


def safe_scenario(n: int = 3) -> dict:
    """Well-separated, low-noise internal scenario for non-regression tests."""
    positions = [(i * 10000.0, i * 20000.0, 5000.0 + i * 250.0) for i in range(n)]
    return {"positions": positions, "resources": [0.9] * n, "id_confidence": {i: 0.95 for i in range(n)}}


def scenario_suite(n: int = 3) -> list[dict]:
    return [boundary_scenario(n), conflict_scenario(n)]


def sample_scenario(n: int = 3, seed: Optional[int] = None, difficulty: float = 0.5) -> dict:
    """Sample a reproducible scenario by rule-relevant knobs, not by blind random scattering."""
    rng = random.Random(seed)
    # Keep sampled cases initially recoverable under the point-mass dynamics.
    # The fixed boundary/conflict templates still exercise genuinely tight
    # geometry; sampled difficulty is spent on uncertainty, resources and
    # communications rather than on unavoidable one-step collisions.
    separation = 1800.0 - 500.0 * max(0.0, min(1.0, difficulty))
    # Place aircraft on a cross-track line.  The environment assigns
    # alternating headings, so this keeps nominal paths separated while
    # retaining close-range and uncertainty stress at larger difficulty.
    positions = [(0.0, 0.0, 5000.0)]
    for i in range(1, n):
        positions.append((rng.uniform(-250, 250), separation * i, 5000.0 + rng.uniform(-200, 200)))
    return {
        "positions": positions,
        "resources": [max(0.1, 0.8 - difficulty * 0.65)] + [rng.uniform(0.55, 0.95) for _ in range(n - 1)],
        "c2_lost": {0: difficulty > 0.75},
        "sensor_degraded": {0: difficulty > 0.45},
        "id_confidence": {0: max(0.35, 0.95 - difficulty * 0.6)},
    }
