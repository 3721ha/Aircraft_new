"""Aircraft adapter for the official MAPPO-Lagrangian implementation."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

from aircraft_sim.models import Action
from aircraft_sim.rl_api import ObservationEncoder
from aircraft_sim.rules import RuleMonitor, is_hard_rule
from aircraft_sim.stl import STLMonitor
from .common import Box, ContinuousAircraftActionCodec


PINNED_COMMIT = "b80a9f5b4a0049125a827be8fb9c477f69ae021b"
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "external_methods" / "inbox" / "macpo" / "SOURCE"


@dataclass
class OfficialMAPPOLagrangianConfig:
    source: Path = DEFAULT_SOURCE
    hidden_size: int = 64
    layer_N: int = 1
    learning_rate: float = 5e-4
    critic_learning_rate: float = 5e-4
    ppo_epoch: int = 5
    num_mini_batch: int = 1
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 10.0
    safety_bound: float = 0.10
    initial_lagrangian_multiplier: float = 0.78
    lagrangian_learning_rate: float = 5e-4
    cost_signal: str = "episode_binary"
    verify_commit: bool = True


def _commit(source: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_official_core(source: Path):
    package_root = (source / "MAPPO-Lagrangian").resolve()
    package_dir = package_root / "mappo_lagrangian"
    trainer_path = package_dir / "algorithms" / "r_mappo" / "r_mappo_lagr.py"
    if not trainer_path.exists():
        raise FileNotFoundError(f"official MAPPO-Lagrangian source not found: {package_root}")
    root_text = str(package_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    # Upstream imports experiment runners (and optional wandb) from its root
    # initializer. Core algorithm imports do not require those dependencies.
    loaded = sys.modules.get("mappo_lagrangian")
    expected_path = str(package_dir)
    if loaded is None:
        package = ModuleType("mappo_lagrangian")
        package.__package__ = "mappo_lagrangian"
        package.__path__ = [expected_path]
        sys.modules["mappo_lagrangian"] = package
    elif expected_path not in [str(path) for path in getattr(loaded, "__path__", [])]:
        raise RuntimeError(f"a different mappo_lagrangian package is already loaded: {loaded!r}")

    from mappo_lagrangian.algorithms.r_mappo.algorithm.MACPPOPolicy import MACPPOPolicy
    from mappo_lagrangian.algorithms.r_mappo.r_mappo_lagr import R_MAPPO_Lagr
    from mappo_lagrangian.utils.separated_buffer import SeparatedReplayBuffer

    return MACPPOPolicy, R_MAPPO_Lagr, SeparatedReplayBuffer


class OfficialMAPPOLagrangianAdapter:
    """Run the pinned separated-policy MAPPO-Lagrangian on Aircraft."""

    def __init__(self, env, config: OfficialMAPPOLagrangianConfig | None = None, seed: int = 0):
        self.env = env
        self.config = config or OfficialMAPPOLagrangianConfig()
        self.source = Path(self.config.source).resolve()
        self.commit = _commit(self.source)
        if self.config.verify_commit and self.commit != PINNED_COMMIT:
            raise RuntimeError(
                f"MAPPO-Lagrangian source commit {self.commit} does not match pinned {PINNED_COMMIT}"
            )
        if self.config.cost_signal not in {"episode_binary", "step_indicator"}:
            raise ValueError("cost_signal must be 'episode_binary' or 'step_indicator'")
        np.random.seed(seed)
        torch.manual_seed(seed)

        policy_type, trainer_type, buffer_type = _load_official_core(self.source)
        self.encoder = ObservationEncoder(env.n, env.cfg.max_speed, env.cfg.max_altitude)
        self.action_codec = ContinuousAircraftActionCodec(
            env.n, semantic_targets=hasattr(env, "task")
        )
        self.obs_space = Box((self.encoder.dimension,))
        self.share_obs_space = Box((self.encoder.dimension * env.n,))
        self.action_space = Box((self.action_codec.dimension,))
        self.args = self._official_args()
        self.policies = [
            policy_type(self.args, self.obs_space, self.share_obs_space, self.action_space, torch.device("cpu"))
            for _ in range(env.n)
        ]
        self.trainers = [
            trainer_type(self.args, policy, device=torch.device("cpu")) for policy in self.policies
        ]
        self.buffer_type = buffer_type
        self.environment_steps = 0
        self.agent_decisions = 0
        self.updates = 0

    def _official_args(self):
        cfg = self.config
        return SimpleNamespace(
            algorithm_name="mappo_lagr",
            lr=cfg.learning_rate,
            critic_lr=cfg.critic_learning_rate,
            opti_eps=1e-5,
            weight_decay=0.0,
            hidden_size=cfg.hidden_size,
            layer_N=cfg.layer_N,
            use_feature_normalization=True,
            use_orthogonal=True,
            use_ReLU=True,
            stacked_frames=1,
            gain=0.01,
            std_x_coef=1.0,
            std_y_coef=0.5,
            use_policy_active_masks=True,
            use_value_active_masks=True,
            use_naive_recurrent_policy=False,
            use_recurrent_policy=False,
            recurrent_N=1,
            use_popart=False,
            use_valuenorm=False,
            clip_param=cfg.clip_param,
            ppo_epoch=cfg.ppo_epoch,
            num_mini_batch=cfg.num_mini_batch,
            data_chunk_length=10,
            value_loss_coef=cfg.value_loss_coef,
            entropy_coef=cfg.entropy_coef,
            max_grad_norm=cfg.max_grad_norm,
            use_max_grad_norm=True,
            use_clipped_value_loss=True,
            use_huber_loss=True,
            huber_delta=10.0,
            episode_length=self.env.cfg.horizon,
            n_rollout_threads=1,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            use_gae=True,
            use_proper_time_limits=False,
            safety_bound=cfg.safety_bound,
            lamda_lagr=cfg.initial_lagrangian_multiplier,
            lagrangian_coef_rate=cfg.lagrangian_learning_rate,
        )

    def _encode_observations(self, observations: dict) -> np.ndarray:
        zero = [0.0] * self.encoder.dimension
        return np.asarray([
            self.encoder.encode(observations[uid], uid) if uid in observations else zero
            for uid in range(self.env.n)
        ], dtype=np.float32)

    @staticmethod
    def _shared_observation(encoded: np.ndarray) -> np.ndarray:
        return encoded.reshape(-1).astype(np.float32)

    def decode_action(self, latent) -> Action:
        return self.action_codec.decode(latent)

    @torch.no_grad()
    def act(self, observation: dict, agent_id: int = 0, deterministic: bool = True) -> Action:
        self.trainers[agent_id].prep_rollout()
        encoded = np.asarray([self.encoder.encode(observation, agent_id)], dtype=np.float32)
        rnn_state = np.zeros((1, 1, self.config.hidden_size), dtype=np.float32)
        mask = np.ones((1, 1), dtype=np.float32)
        latent, _ = self.policies[agent_id].act(encoded, rnn_state, mask, deterministic=deterministic)
        return self.decode_action(latent.detach().cpu().numpy()[0])

    @staticmethod
    def _scalar(value):
        if torch.is_tensor(value):
            return float(value.detach().float().mean().cpu())
        return float(np.asarray(value, dtype=float).mean())

    def train_episode(self, scenario: dict | None = None) -> dict:
        observations = self.env.reset(scenario)
        monitor = RuleMonitor(self.env.cfg)
        encoded = self._encode_observations(observations)
        shared = self._shared_observation(encoded)
        buffers = [
            self.buffer_type(self.args, self.obs_space, self.share_obs_space, self.action_space)
            for _ in range(self.env.n)
        ]
        for uid, buffer in enumerate(buffers):
            buffer.obs[0, 0] = encoded[uid]
            buffer.share_obs[0, 0] = shared

        episode_reward = 0.0
        step_cost_sum = 0.0
        truths = []
        actions_history = []
        steps_collected = 0
        for _ in range(self.args.episode_length):
            collected = []
            for uid in range(self.env.n):
                trainer, policy, buffer = self.trainers[uid], self.policies[uid], buffers[uid]
                trainer.prep_rollout()
                with torch.no_grad():
                    result = policy.get_actions(
                        buffer.share_obs[buffer.step],
                        buffer.obs[buffer.step],
                        buffer.rnn_states[buffer.step],
                        buffer.rnn_states_critic[buffer.step],
                        buffer.masks[buffer.step],
                        rnn_states_cost=buffer.rnn_states_cost[buffer.step],
                    )
                collected.append([item.detach().cpu().numpy() for item in result])

            action_dict = {uid: self.decode_action(collected[uid][1][0]) for uid in range(self.env.n)}
            step = self.env.step(action_dict)
            episode_reward += float(step.reward)
            truth_report = monitor.evaluate_truth(step.info["truth"], action_dict)
            team_cost = float(any(is_hard_rule(rule) for rule in truth_report.violations))
            step_cost_sum += team_cost
            truths.append(step.info["truth"])
            actions_history.append(action_dict)
            next_encoded = self._encode_observations(step.observations)
            next_shared = self._shared_observation(next_encoded)
            mask = np.zeros((1, 1), dtype=np.float32) if step.done else np.ones((1, 1), dtype=np.float32)
            for uid, buffer in enumerate(buffers):
                values, latent, log_prob, rnn_actor, rnn_critic, cost_pred, rnn_cost = collected[uid]
                buffer.insert(
                    next_shared[None, :],
                    next_encoded[uid][None, :],
                    rnn_actor,
                    rnn_critic,
                    latent,
                    log_prob,
                    values,
                    np.asarray([[step.reward]], dtype=np.float32),
                    mask,
                    costs=np.asarray([[team_cost]], dtype=np.float32),
                    cost_preds=cost_pred,
                    rnn_states_cost=rnn_cost,
                )
            observations = step.observations
            self.environment_steps += 1
            self.agent_decisions += self.env.n
            steps_collected += 1
            if step.done:
                break

        trajectory_cost = float(
            not STLMonitor(RuleMonitor(self.env.cfg)).evaluate(truths, actions_history).joint_hard_satisfied
        )
        if self.config.cost_signal == "episode_binary":
            for buffer in buffers:
                buffer.costs[:steps_collected] = 0.0
                buffer.costs[steps_collected - 1, 0, 0] = trajectory_cost
            constraint_cost = trajectory_cost
        else:
            constraint_cost = step_cost_sum

        for uid, buffer in enumerate(buffers):
            trainer, policy = self.trainers[uid], self.policies[uid]
            trainer.prep_rollout()
            with torch.no_grad():
                next_value = policy.get_values(
                    buffer.share_obs[-1], buffer.rnn_states_critic[-1], buffer.masks[-1]
                ).detach().cpu().numpy()
                next_cost = policy.get_cost_values(
                    buffer.share_obs[-1], buffer.rnn_states_cost[-1], buffer.masks[-1]
                ).detach().cpu().numpy()
            buffer.compute_returns(next_value, trainer.value_normalizer)
            buffer.compute_cost_returns(next_cost, trainer.value_normalizer)
            buffer.return_aver_insert(np.asarray([constraint_cost], dtype=np.float32))

        factor = np.ones((self.args.episode_length, 1, 5), dtype=np.float32)
        per_agent = []
        for uid in torch.randperm(self.env.n).tolist():
            trainer, policy, buffer = self.trainers[uid], self.policies[uid], buffers[uid]
            trainer.prep_rollout()
            with torch.no_grad():
                old_log_prob, _ = policy.actor.evaluate_actions(
                    buffer.obs[:-1].reshape(-1, self.encoder.dimension),
                    buffer.rnn_states[0:1].reshape(-1, 1, self.config.hidden_size),
                    buffer.actions.reshape(-1, self.action_codec.dimension),
                    buffer.masks[:-1].reshape(-1, 1),
                    None,
                    buffer.active_masks[:-1].reshape(-1, 1),
                )
            buffer.update_factor(factor)
            trainer.prep_training()
            stats = trainer.train(buffer)
            trainer.prep_rollout()
            with torch.no_grad():
                new_log_prob, _ = policy.actor.evaluate_actions(
                    buffer.obs[:-1].reshape(-1, self.encoder.dimension),
                    buffer.rnn_states[0:1].reshape(-1, 1, self.config.hidden_size),
                    buffer.actions.reshape(-1, self.action_codec.dimension),
                    buffer.masks[:-1].reshape(-1, 1),
                    None,
                    buffer.active_masks[:-1].reshape(-1, 1),
                )
            factor *= np.exp(
                (new_log_prob - old_log_prob).detach().cpu().numpy().reshape(self.args.episode_length, 1, 5)
            )
            per_agent.append({
                "agent": uid,
                "lagrangian_multiplier": self._scalar(trainer.lamda_lagr),
                **{key: self._scalar(value) for key, value in stats.items()},
            })
            buffer.after_update()

        self.updates += 1
        return {
            "episode_reward": episode_reward,
            "episode_hard_violation_cost": trajectory_cost,
            "mean_step_team_cost": step_cost_sum / max(1, steps_collected),
            "constraint_cost": constraint_cost,
            "cost_signal": self.config.cost_signal,
            "environment_steps": self.environment_steps,
            "agent_decisions": self.agent_decisions,
            "updates": self.updates,
            "agents": per_agent,
        }

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        multipliers = []
        for uid, (policy, trainer) in enumerate(zip(self.policies, self.trainers)):
            torch.save(policy.actor.state_dict(), directory / f"actor_{uid}.pt")
            torch.save(policy.critic.state_dict(), directory / f"critic_{uid}.pt")
            torch.save(policy.cost_critic.state_dict(), directory / f"cost_critic_{uid}.pt")
            multipliers.append(self._scalar(trainer.lamda_lagr))
        (directory / "lagrangian_multipliers.json").write_text(
            json.dumps(multipliers, indent=2), encoding="utf-8"
        )
        (directory / "ADAPTER_METADATA.txt").write_text(
            "\n".join([
                "repository=https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation",
                f"commit={self.commit}",
                "component=MAPPO-Lagrangian",
                "action_adapter=continuous latent Box(5) with joint semantic target decode",
                f"cost_signal={self.config.cost_signal}",
                "",
            ]),
            encoding="utf-8",
        )

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        multipliers_path = directory / "lagrangian_multipliers.json"
        multipliers = json.loads(multipliers_path.read_text(encoding="utf-8")) if multipliers_path.exists() else None
        for uid, (policy, trainer) in enumerate(zip(self.policies, self.trainers)):
            policy.actor.load_state_dict(
                torch.load(directory / f"actor_{uid}.pt", map_location="cpu", weights_only=True)
            )
            policy.critic.load_state_dict(
                torch.load(directory / f"critic_{uid}.pt", map_location="cpu", weights_only=True)
            )
            policy.cost_critic.load_state_dict(
                torch.load(directory / f"cost_critic_{uid}.pt", map_location="cpu", weights_only=True)
            )
            if multipliers is not None:
                trainer.lamda_lagr = float(multipliers[uid])

    @property
    def source_fingerprint(self) -> str:
        return hashlib.sha256(
            self.commit.encode("ascii") + (self.source / "LICENSE").read_bytes()
        ).hexdigest()
