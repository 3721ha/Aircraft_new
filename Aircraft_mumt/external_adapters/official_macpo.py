"""Aircraft adapter for chauncygu's official MACPO implementation."""

from dataclasses import dataclass
import hashlib
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
class OfficialMACPOConfig:
    source: Path = DEFAULT_SOURCE
    hidden_size: int = 64
    layer_N: int = 1
    learning_rate: float = 5e-4
    critic_learning_rate: float = 5e-4
    ppo_epoch: int = 1
    num_mini_batch: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    safety_bound: float = 0.10
    kl_threshold: float = 0.01
    line_search_steps: int = 10
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
    package_root = (source / "MACPO").resolve()
    package_dir = package_root / "macpo"
    if not (package_dir / "algorithms" / "r_mappo" / "r_macpo.py").exists():
        raise FileNotFoundError(f"official MACPO source not found: {package_root}")
    root_text = str(package_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    # The upstream package initializer eagerly imports its experiment runner,
    # which makes optional logging dependencies mandatory even for core MACPO.
    # Registering the package path preserves normal submodule imports without
    # executing that initializer or modifying the pinned third-party source.
    loaded = sys.modules.get("macpo")
    expected_path = str(package_dir)
    if loaded is None:
        package = ModuleType("macpo")
        package.__package__ = "macpo"
        package.__path__ = [expected_path]
        sys.modules["macpo"] = package
    elif expected_path not in [str(path) for path in getattr(loaded, "__path__", [])]:
        raise RuntimeError(f"a different macpo package is already loaded: {loaded!r}")

    from macpo.algorithms.r_mappo.algorithm.MACPPOPolicy import MACPPOPolicy
    from macpo.algorithms.r_mappo.r_macpo import R_MACTRPO_CPO
    from macpo.utils.separated_buffer import SeparatedReplayBuffer

    return MACPPOPolicy, R_MACTRPO_CPO, SeparatedReplayBuffer


class OfficialMACPOAdapter:
    """Run the pinned separated-policy MACPO algorithm on Aircraft."""

    def __init__(self, env, config: OfficialMACPOConfig | None = None, seed: int = 0):
        self.env = env
        self.config = config or OfficialMACPOConfig()
        self.source = Path(self.config.source).resolve()
        self.commit = _commit(self.source)
        if self.config.verify_commit and self.commit != PINNED_COMMIT:
            raise RuntimeError(f"MACPO source commit {self.commit} does not match pinned {PINNED_COMMIT}")
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
        self.trainers = [trainer_type(self.args, policy, device=torch.device("cpu")) for policy in self.policies]
        self.buffer_type = buffer_type
        self.environment_steps = 0
        self.agent_decisions = 0
        self.updates = 0

    def _official_args(self):
        cfg = self.config
        return SimpleNamespace(
            algorithm_name="macpo",
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
            clip_param=0.2,
            ppo_epoch=cfg.ppo_epoch,
            num_mini_batch=cfg.num_mini_batch,
            data_chunk_length=10,
            value_loss_coef=1.0,
            entropy_coef=0.01,
            max_grad_norm=10.0,
            use_max_grad_norm=True,
            use_clipped_value_loss=True,
            use_huber_loss=True,
            huber_delta=10.0,
            episode_length=self.env.cfg.horizon,
            n_rollout_threads=1,
            gamma=cfg.gamma,
            safety_gamma=0.2,
            gae_lambda=cfg.gae_lambda,
            use_gae=True,
            use_proper_time_limits=False,
            kl_threshold=cfg.kl_threshold,
            safety_bound=cfg.safety_bound,
            ls_step=cfg.line_search_steps,
            accept_ratio=0.5,
            EPS=1e-8,
            line_search_fraction=0.5,
            g_step_dir_coef=0.1,
            b_step_dir_coef=0.1,
            fraction_coef=0.1,
        )

    def _encode_observations(self, observations: dict) -> np.ndarray:
        zero = [0.0] * self.encoder.dimension
        return np.asarray([
            self.encoder.encode(observations[uid], uid) if uid in observations else zero
            for uid in range(self.env.n)
        ], dtype=np.float32)

    def _shared_observation(self, encoded: np.ndarray) -> np.ndarray:
        return encoded.reshape(-1).astype(np.float32)

    def decode_action(self, latent) -> Action:
        """Map the official unbounded Gaussian latent into Aircraft actions."""

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
        """Collect one episode and execute the official sequential CPO update."""

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
        episode_costs = np.zeros(self.env.n, dtype=np.float32)
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
            episode_costs += team_cost
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

        trajectory_cost = float(not STLMonitor(RuleMonitor(self.env.cfg)).evaluate(truths, actions_history).joint_hard_satisfied)
        if self.config.cost_signal == "episode_binary":
            for buffer in buffers:
                buffer.costs[:steps_collected] = 0.0
                buffer.costs[steps_collected - 1, 0, 0] = trajectory_cost
            constraint_cost = trajectory_cost
        else:
            constraint_cost = float(episode_costs.mean())

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
            # Upstream copies this value, so pass a NumPy scalar array rather
            # than a Python float. The trainer only consumes its mean.
            episode_cost = np.asarray([constraint_cost], dtype=np.float32)
            buffer.return_aver_insert(episode_cost)

        factor = np.ones((self.args.episode_length, 1, 5), dtype=np.float32)
        per_agent = []
        for uid in torch.randperm(self.env.n).tolist():
            trainer, policy, buffer = self.trainers[uid], self.policies[uid], buffers[uid]
            trainer.prep_rollout()
            with torch.no_grad():
                old_log_prob, _, _, _ = policy.actor.evaluate_actions(
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
                new_log_prob, _, _, _ = policy.actor.evaluate_actions(
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
                **{key: self._scalar(value) for key, value in stats.items()},
            })
            buffer.after_update()

        self.updates += 1
        return {
            "episode_reward": episode_reward,
            "episode_hard_violation_cost": trajectory_cost,
            "mean_step_team_cost": float(episode_costs.mean() / max(1, self.args.episode_length)),
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
        for uid, policy in enumerate(self.policies):
            torch.save(policy.actor.state_dict(), directory / f"actor_{uid}.pt")
            torch.save(policy.critic.state_dict(), directory / f"critic_{uid}.pt")
            torch.save(policy.cost_critic.state_dict(), directory / f"cost_critic_{uid}.pt")
        (directory / "ADAPTER_METADATA.txt").write_text(
            "\n".join([
                "repository=https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation",
                f"commit={self.commit}",
                "action_adapter=continuous latent Box(5) with joint semantic target decode",
                "training_cost=team hard-rule violation indicator",
                f"cost_signal={self.config.cost_signal}",
                "",
            ]),
            encoding="utf-8",
        )

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        for uid, policy in enumerate(self.policies):
            policy.actor.load_state_dict(torch.load(directory / f"actor_{uid}.pt", map_location="cpu", weights_only=True))
            policy.critic.load_state_dict(torch.load(directory / f"critic_{uid}.pt", map_location="cpu", weights_only=True))
            policy.cost_critic.load_state_dict(torch.load(directory / f"cost_critic_{uid}.pt", map_location="cpu", weights_only=True))

    @property
    def source_fingerprint(self) -> str:
        return hashlib.sha256(self.commit.encode("ascii") + (self.source / "LICENSE").read_bytes()).hexdigest()
