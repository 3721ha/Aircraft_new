"""Shared spaces and Aircraft action semantics for external algorithms."""

from typing import Sequence

import numpy as np

from aircraft_sim.models import Action, TargetKind
from aircraft_sim.rl_api import ActionCodec


class Box:
    """Minimal legacy-Gym-compatible shape object."""

    def __init__(self, shape, low=-1.0, high=1.0):
        self.shape = tuple(shape)
        self.low = np.full(self.shape, low, dtype=np.float32)
        self.high = np.full(self.shape, high, dtype=np.float32)


class MultiDiscrete:
    """Minimal legacy action-space metadata; bounds are inclusive."""

    def __init__(self, high: Sequence[int]):
        self.low = np.zeros(len(high), dtype=np.int64)
        self.high = np.asarray(high, dtype=np.int64)
        self.shape = len(high)


class ContinuousAircraftActionCodec:
    """Decode a common continuous latent action used by external baselines.

    Coordinates 0 and 4 represent semantic mode and target assignment.  In
    active-task environments the final coordinate jointly encodes no target,
    a teammate id, the intruder, the protected asset, or a shared resource.
    This preserves the five-dimensional action interface assumed by the
    pinned official algorithms while exposing the same semantic target set.
    They are clipped and rounded only at the simulator boundary. Coordinates
    1--3 directly represent normalized turn, climb and acceleration commands.
    """

    dimension = 5

    def __init__(self, n_aircraft: int, semantic_targets: bool = False):
        self.n_aircraft = n_aircraft
        self.semantic_targets = bool(semantic_targets)
        self.codec = ActionCodec(n_aircraft)

    def decode(self, latent: Sequence[float]) -> Action:
        raw = np.asarray(latent, dtype=float).reshape(-1)
        if len(raw) != self.dimension:
            raise ValueError("continuous Aircraft action must contain five coordinates")
        bounded = np.clip(raw, -1.0, 1.0)
        mode = (bounded[0] + 1.0) * (len(self.codec.mode_order) - 1) / 2.0
        if not self.semantic_targets:
            target = (bounded[4] + 1.0) * self.n_aircraft / 2.0 - 1.0
            return self.codec.decode([mode, bounded[1], bounded[2], bounded[3], target])
        # Choices: 0=no target, 1..N=teammate, N+1=intruder,
        # N+2=protected asset, N+3=shared resource.
        choice = int(round((bounded[4] + 1.0) * (self.n_aircraft + 3) / 2.0))
        if choice <= 0:
            target, target_kind = None, TargetKind.NONE
        elif choice <= self.n_aircraft:
            target, target_kind = choice - 1, TargetKind.TEAMMATE
        elif choice == self.n_aircraft + 1:
            target, target_kind = None, TargetKind.INTRUDER
        elif choice == self.n_aircraft + 2:
            target, target_kind = None, TargetKind.PROTECTED_ASSET
        else:
            target, target_kind = None, TargetKind.RESOURCE
        return Action(
            self.codec.mode_order[max(0, min(len(self.codec.mode_order) - 1, int(round(mode))))],
            float(bounded[1]), float(bounded[2]), float(bounded[3]),
            target, target_kind,
        ).clipped()
