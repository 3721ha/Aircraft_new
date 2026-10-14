"""Higher-fidelity validation adapter with the same decision-level API.

The research environment deliberately keeps the action, observation, rule and
logging contracts independent from the vehicle dynamics.  This module provides
an inexpensive transfer test with actuator lag and rate limits.  It is a
validation proxy, not a six-degree-of-freedom flight model.  A JSBSim backend
can be selected when the optional dependency is installed; the factory fails
explicitly instead of silently substituting the proxy.
"""

from __future__ import annotations

import importlib.util
from typing import Dict, Optional, Tuple

from .config import SimConfig
from .env import AircraftEnv
from .models import Action


def jsbsim_available() -> bool:
    """Return whether the optional JSBSim Python package is importable."""

    return importlib.util.find_spec("jsbsim") is not None


class HigherFidelityAircraftEnv(AircraftEnv):
    """Decision-compatible dynamics proxy for cross-model validation.

    Compared with :class:`AircraftEnv`, commanded turn, climb and acceleration
    are filtered by a first-order actuator model and bounded by rate limits.
    This exposes controller brittleness caused by delayed execution while
    preserving exactly the same semantic action and observation interfaces.
    """

    backend_name = "lagged_3dof_validation_proxy"

    def __init__(
        self,
        n_aircraft: int = 3,
        config: Optional[SimConfig] = None,
        *,
        actuator_tau: float = 0.45,
        max_turn_rate: float = 0.55,
        max_climb_rate: float = 0.40,
        max_acceleration_rate: float = 0.75,
        randomize_dynamics: bool = False,
        actuator_tau_range: Tuple[float, float] = (0.30, 0.70),
        rate_scale_range: Tuple[float, float] = (0.80, 1.20),
    ):
        super().__init__(n_aircraft=n_aircraft, config=config)
        if actuator_tau <= 0:
            raise ValueError("actuator_tau must be positive")
        self.actuator_tau = float(actuator_tau)
        self.max_turn_rate = float(max_turn_rate)
        self.max_climb_rate = float(max_climb_rate)
        self.max_acceleration_rate = float(max_acceleration_rate)
        self.randomize_dynamics = bool(randomize_dynamics)
        if actuator_tau_range[0] <= 0 or actuator_tau_range[1] < actuator_tau_range[0]:
            raise ValueError("actuator_tau_range must be positive and ordered")
        if rate_scale_range[0] <= 0 or rate_scale_range[1] < rate_scale_range[0]:
            raise ValueError("rate_scale_range must be positive and ordered")
        self.actuator_tau_range = tuple(float(value) for value in actuator_tau_range)
        self.rate_scale_range = tuple(float(value) for value in rate_scale_range)
        self._episode_actuator_tau = self.actuator_tau
        self._episode_rate_scale = 1.0
        self._applied_controls: Dict[int, Tuple[float, float, float]] = {}

    @property
    def dynamics_metadata(self) -> dict:
        return {
            "backend": self.backend_name,
            "actuator_tau": self.actuator_tau,
            "max_turn_rate": self.max_turn_rate,
            "max_climb_rate": self.max_climb_rate,
            "max_acceleration_rate": self.max_acceleration_rate,
            "randomize_dynamics": self.randomize_dynamics,
            "episode_actuator_tau": self._episode_actuator_tau,
            "episode_rate_scale": self._episode_rate_scale,
            "truth_access_for_policy": False,
        }

    def reset(self, scenario: Optional[dict] = None) -> dict:
        observations = super().reset(scenario)
        if self.randomize_dynamics:
            self._episode_actuator_tau = self.rng.uniform(*self.actuator_tau_range)
            self._episode_rate_scale = self.rng.uniform(*self.rate_scale_range)
        else:
            self._episode_actuator_tau = self.actuator_tau
            self._episode_rate_scale = 1.0
        self._applied_controls = {i: (0.0, 0.0, 0.0) for i in range(self.n)}
        return observations

    @staticmethod
    def _rate_limited(previous: float, command: float, alpha: float, rate: float, dt: float) -> float:
        target = previous + alpha * (command - previous)
        delta = max(-rate * dt, min(rate * dt, target - previous))
        return max(-1.0, min(1.0, previous + delta))

    def step(self, actions: Dict[int, Action]):
        alpha = min(1.0, self.cfg.dt / self._episode_actuator_tau)
        applied: Dict[int, Action] = {}
        for i in range(self.n):
            command = actions.get(i, Action()).clipped()
            previous = self._applied_controls.get(i, (0.0, 0.0, 0.0))
            controls = (
                self._rate_limited(previous[0], command.turn, alpha, self.max_turn_rate * self._episode_rate_scale, self.cfg.dt),
                self._rate_limited(previous[1], command.climb, alpha, self.max_climb_rate * self._episode_rate_scale, self.cfg.dt),
                self._rate_limited(previous[2], command.acceleration, alpha, self.max_acceleration_rate * self._episode_rate_scale, self.cfg.dt),
            )
            self._applied_controls[i] = controls
            applied[i] = Action(
                command.mode, controls[0], controls[1], controls[2],
                command.target, command.target_kind,
            )
        result = super().step(applied)
        result.info["dynamics"] = self.dynamics_metadata
        result.info["commanded_actions"] = {i: actions.get(i, Action()).clipped() for i in range(self.n)}
        result.info["applied_actions"] = applied
        return result


def make_validation_env(
    backend: str = "surrogate",
    n_aircraft: int = 3,
    config: Optional[SimConfig] = None,
    **kwargs,
) -> AircraftEnv:
    """Construct a validation backend with an explicit dependency check."""

    normalized = backend.lower().strip()
    if normalized in {"surrogate", "proxy", "lagged_3dof"}:
        return HigherFidelityAircraftEnv(n_aircraft, config, **kwargs)
    if normalized == "jsbsim":
        if not jsbsim_available():
            raise RuntimeError(
                "JSBSim backend requested but the optional 'jsbsim' package is "
                "not installed. Install it separately, then rerun with --backend jsbsim."
            )
        from .jsbsim_env import JSBSimAircraftEnv
        return JSBSimAircraftEnv(n_aircraft, config, **kwargs)
    raise ValueError(f"unknown validation backend: {backend}")
