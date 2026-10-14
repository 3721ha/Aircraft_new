"""Physical-control bridge for MIT-REALM's official GCBF+ implementation."""

from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from aircraft_sim.models import Action
from aircraft_sim.rules import RuleMonitor
from aircraft_sim.shield import Intervention, SafetyShield


PINNED_COMMIT = "fb449907bdbf981aa10f0edfecca02663ddc8037"
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "external_methods" / "inbox" / "gcbfplus" / "SOURCE"

REQUIRED_MODULES = ("jax", "flax", "jraph", "optax", "tensorflow_probability", "jaxproxqp")


class GCBFPlusDependencyError(RuntimeError):
    pass


@dataclass
class OfficialGCBFPlusConfig:
    source: Path = DEFAULT_SOURCE
    checkpoint_step: int = 1000
    area_size: float = 2.0
    pretrained_agents: int = 8
    checkpoint_obstacles: int = 8
    deployed_obstacles: int = 0
    lidar_rays: int = 32
    seed: int = 2
    verify_commit: bool = True


def dependencies_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES)


def _commit(source: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class OfficialGCBFPlusShield:
    """Use pretrained LinearDrone GCBF+ to replace nominal physical controls.

    Semantic mode and target decisions remain with the nominal Aircraft policy.
    The bridge is intentionally limited to the S01/S02/S03 physical subtask.
    """

    physical_rule_scope = ("S01", "S02", "S03")

    def __init__(self, env, config: OfficialGCBFPlusConfig | None = None):
        self.env = env
        self.config = config or OfficialGCBFPlusConfig()
        self.source = Path(self.config.source).resolve()
        self.commit = _commit(self.source)
        if self.config.verify_commit and self.commit != PINNED_COMMIT:
            raise RuntimeError(f"GCBF+ source commit {self.commit} does not match pinned {PINNED_COMMIT}")
        missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
        if missing:
            raise GCBFPlusDependencyError(
                "official GCBF+ requires an isolated Python 3.10 environment; missing: " + ", ".join(missing)
            )

        source_text = str(self.source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        import jax.numpy as jnp
        import jax.random as jr
        from gcbfplus.algo import make_algo
        from gcbfplus.env import make_env

        self.jnp = jnp
        self.jr = jr
        self.monitor = RuleMonitor(env.cfg)
        self.gcbf_env = make_env(
            env_id="LinearDrone",
            num_agents=env.n,
            num_obs=self.config.deployed_obstacles,
            n_rays=self.config.lidar_rays,
            area_size=self.config.area_size,
            max_step=env.cfg.horizon,
        )
        self.algo = make_algo(
            algo="gcbf+",
            env=self.gcbf_env,
            node_dim=self.gcbf_env.node_dim,
            edge_dim=self.gcbf_env.edge_dim,
            state_dim=self.gcbf_env.state_dim,
            action_dim=self.gcbf_env.action_dim,
            n_agents=env.n,
            gnn_layers=1,
            batch_size=256,
            buffer_size=512,
            horizon=32,
            lr_actor=1e-5,
            lr_cbf=1e-5,
            alpha=1.0,
            eps=0.02,
            inner_epoch=8,
            loss_action_coef=1e-4,
            loss_unsafe_coef=1.0,
            loss_safe_coef=1.0,
            loss_h_dot_coef=0.01,
            max_grad_norm=2.0,
            seed=self.config.seed,
        )
        checkpoint_dir = self.source / "pretrained" / "LinearDrone" / "gcbf+" / "models"
        self.algo.load(str(checkpoint_dir), self.config.checkpoint_step)
        self.template_graph = self.gcbf_env.reset(self.jr.PRNGKey(self.config.seed))
        self.position_scale = env.cfg.min_separation / (2.0 * self.gcbf_env.params["drone_radius"])
        self.last_raw_action = None

    def _graph(self, observations: dict, nominal: dict[int, Action]):
        positions = np.asarray([observations[uid]["self"]["position"] for uid in range(self.env.n)], dtype=float)
        centered = positions - positions.mean(axis=0, keepdims=True)
        normalized_position = centered / self.position_scale + self.config.area_size / 2.0

        velocities = []
        goals = []
        for uid in range(self.env.n):
            own = observations[uid]["self"]
            heading = float(own.get("heading", 0.0))
            pitch = float(own.get("pitch", 0.0))
            speed = float(own.get("speed", 0.0)) / max(1.0, self.env.cfg.max_speed) * 0.4
            velocities.append([
                speed * math.cos(pitch) * math.cos(heading),
                speed * math.cos(pitch) * math.sin(heading),
                speed * math.sin(pitch),
            ])
            action = nominal[uid]
            desired_heading = heading + 0.4 * action.turn
            desired_pitch = max(-0.45, min(0.45, pitch + 0.2 * action.climb))
            direction = np.asarray([
                math.cos(desired_pitch) * math.cos(desired_heading),
                math.cos(desired_pitch) * math.sin(desired_heading),
                math.sin(desired_pitch),
            ])
            goals.append(normalized_position[uid] + 0.35 * direction)

        agent_state = np.concatenate([normalized_position, np.asarray(velocities)], axis=1)
        goal_state = np.concatenate([np.asarray(goals), np.zeros((self.env.n, 3))], axis=1)
        env_state = self.template_graph.env_states._replace(
            agent=self.jnp.asarray(agent_state, dtype=self.jnp.float32),
            goal=self.jnp.asarray(goal_state, dtype=self.jnp.float32),
        )
        return self.gcbf_env.get_graph(env_state)

    @staticmethod
    def map_physical_control(raw_action, heading: float) -> tuple[float, float, float]:
        ax, ay, az = [float(value) for value in np.asarray(raw_action).reshape(3)]
        forward = ax * math.cos(heading) + ay * math.sin(heading)
        lateral = -ax * math.sin(heading) + ay * math.cos(heading)
        return tuple(float(np.clip(value, -1.0, 1.0)) for value in (lateral, az, forward))

    def filter(self, observations: dict, nominal: dict[int, Action]):
        graph = self._graph(observations, nominal)
        raw_actions = np.asarray(self.algo.act(graph), dtype=float)
        self.last_raw_action = raw_actions
        safe = {}
        interventions = []
        for uid, original in nominal.items():
            heading = float(observations[uid]["self"].get("heading", 0.0))
            turn, climb, acceleration = self.map_physical_control(raw_actions[uid], heading)
            corrected = Action(
                mode=original.mode,
                turn=turn,
                climb=climb,
                acceleration=acceleration,
                target=original.target,
            ).clipped()
            safe[uid] = corrected
            distance = SafetyShield._distance(original, corrected)
            if distance > 1e-9:
                interventions.append(
                    Intervention(uid, original, corrected, ["GCBF+ physical control"], distance)
                )
        return safe, interventions

    @property
    def source_fingerprint(self) -> str:
        return hashlib.sha256(
            self.commit.encode("ascii") + (self.source / "LICENSE").read_bytes()
        ).hexdigest()
