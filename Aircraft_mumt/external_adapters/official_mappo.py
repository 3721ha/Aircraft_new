"""Aircraft adapter for marlbenchmark/on-policy's official MAPPO core.

The third-party source tree is imported without modification. The formal
comparison uses the common continuous Aircraft latent action; the original
MultiDiscrete grid remains available for adapter-sensitivity experiments.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch

from aircraft_sim.models import Action
from aircraft_sim.rl_api import ActionCodec, ObservationEncoder
from .common import Box, ContinuousAircraftActionCodec, MultiDiscrete


PINNED_COMMIT = "de66d7a4b23fac2513f56f96f73b3f5cb96695ac"
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "external_methods" / "inbox" / "mappo" / "SOURCE"


@dataclass
class OfficialMAPPOConfig:
    source: Path = DEFAULT_SOURCE
    action_adapter: str = "continuous"
    control_bins: int = 11
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
    source = source.resolve()
    if not (source / "onpolicy" / "algorithms" / "r_mappo").is_dir():
        raise FileNotFoundError(f"official MAPPO source not found: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
    from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
    from onpolicy.utils.shared_buffer import SharedReplayBuffer

    return R_MAPPOPolicy, R_MAPPO, SharedReplayBuffer


class OfficialMAPPOAdapter:
    """Use the pinned official MAPPO actor, critic, PPO trainer, and buffer."""

    def __init__(self, env, config: OfficialMAPPOConfig | None = None, seed: int = 0):
        self.env = env
        self.config = config or OfficialMAPPOConfig()
        self.source = Path(self.config.source).resolve()
        self.commit = _commit(self.source)
        if self.config.verify_commit and self.commit != PINNED_COMMIT:
            raise RuntimeError(f"MAPPO source commit {self.commit} does not match pinned {PINNED_COMMIT}")
        if self.config.control_bins < 3 or self.config.control_bins % 2 == 0:
            raise ValueError("control_bins must be an odd integer >= 3")
        if self.config.action_adapter not in {"continuous", "multidiscrete"}:
            raise ValueError("action_adapter must be 'continuous' or 'multidiscrete'")

        np.random.seed(seed)
        torch.manual_seed(seed)
        policy_type, trainer_type, buffer_type = _load_official_core(self.source)
        self.encoder = ObservationEncoder(env.n, env.cfg.max_speed, env.cfg.max_altitude)
        self.codec = ActionCodec(env.n)
        self.continuous_codec = ContinuousAircraftActionCodec(
            env.n, semantic_targets=hasattr(env, "task")
        )
        self.obs_space = Box((self.encoder.dimension,))
        self.share_obs_space = Box((self.encoder.dimension * env.n,))
        if self.config.action_adapter == "continuous":
            self.action_space = Box((self.continuous_codec.dimension,))
        else:
            # mode, turn, climb, acceleration, target assignment
            self.action_space = MultiDiscrete([
                len(self.codec.mode_order) - 1,
                self.config.control_bins - 1,
                self.config.control_bins - 1,
                self.config.control_bins - 1,
                env.n,
            ])
        self.args = self._official_args()
        self.policy = policy_type(self.args, self.obs_space, self.share_obs_space, self.action_space, torch.device("cpu"))
        self.trainer = trainer_type(self.args, self.policy, torch.device("cpu"))
        self.buffer_type = buffer_type
        self.environment_steps = 0
        self.agent_decisions = 0
        self.updates = 0

    def _official_args(self):
        cfg = self.config
        return SimpleNamespace(
            algorithm_name="mappo",
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
        )

    def _encode_observations(self, observations: dict) -> np.ndarray:
        zero = [0.0] * self.encoder.dimension
        return np.asarray(
            [self.encoder.encode(observations[uid], uid) if uid in observations else zero for uid in range(self.env.n)],
            dtype=np.float32,
        )

    def _shared_observations(self, encoded: np.ndarray) -> np.ndarray:
        flattened = encoded.reshape(-1)
        return np.repeat(flattened[None, :], self.env.n, axis=0).astype(np.float32)

    def decode_action(self, action: Sequence[float]) -> Action:
        if self.config.action_adapter == "continuous":
            return self.continuous_codec.decode(action)
        indices = np.asarray(action, dtype=np.int64).reshape(-1)
        if len(indices) != 5:
            raise ValueError("official MAPPO action must contain five discrete indices")
        bins = self.config.control_bins

        def control(index):
            return -1.0 + 2.0 * max(0, min(bins - 1, int(index))) / (bins - 1)

        target = int(indices[4]) - 1
        return self.codec.decode([
            int(indices[0]),
            control(indices[1]),
            control(indices[2]),
            control(indices[3]),
            target,
        ])

    @torch.no_grad()
    def act(self, observation: dict, agent_id: int = 0, deterministic: bool = True) -> Action:
        self.trainer.prep_rollout()
        encoded = np.asarray([self.encoder.encode(observation, agent_id)], dtype=np.float32)
        rnn_state = np.zeros((1, 1, self.config.hidden_size), dtype=np.float32)
        mask = np.ones((1, 1), dtype=np.float32)
        action, _ = self.policy.act(encoded, rnn_state, mask, deterministic=deterministic)
        return self.decode_action(action.detach().cpu().numpy()[0])

    def train_episode(self, scenario: dict | None = None) -> dict:
        """Collect one fixed-length episode and run one official PPO update."""

        observations = self.env.reset(scenario)
        encoded = self._encode_observations(observations)
        shared = self._shared_observations(encoded)
        buffer = self.buffer_type(self.args, self.env.n, self.obs_space, self.share_obs_space, self.action_space)
        buffer.obs[0, 0] = encoded
        buffer.share_obs[0, 0] = shared
        episode_reward = 0.0

        for _ in range(self.args.episode_length):
            self.trainer.prep_rollout()
            with torch.no_grad():
                values, actions, log_probs, rnn_actor, rnn_critic = self.policy.get_actions(
                    buffer.share_obs[buffer.step, 0],
                    buffer.obs[buffer.step, 0],
                    buffer.rnn_states[buffer.step, 0],
                    buffer.rnn_states_critic[buffer.step, 0],
                    buffer.masks[buffer.step, 0],
                )
            values_np = values.detach().cpu().numpy()[None, ...]
            actions_np = actions.detach().cpu().numpy()[None, ...]
            log_probs_np = log_probs.detach().cpu().numpy()[None, ...]
            rnn_actor_np = rnn_actor.detach().cpu().numpy()[None, ...]
            rnn_critic_np = rnn_critic.detach().cpu().numpy()[None, ...]
            action_dict = {uid: self.decode_action(actions_np[0, uid]) for uid in range(self.env.n)}
            result = self.env.step(action_dict)
            episode_reward += float(result.reward)
            next_encoded = self._encode_observations(result.observations)
            next_shared = self._shared_observations(next_encoded)
            rewards = np.full((1, self.env.n, 1), float(result.reward), dtype=np.float32)
            masks = np.zeros((1, self.env.n, 1), dtype=np.float32) if result.done else np.ones((1, self.env.n, 1), dtype=np.float32)
            buffer.insert(
                next_shared[None, ...],
                next_encoded[None, ...],
                rnn_actor_np,
                rnn_critic_np,
                actions_np,
                log_probs_np,
                values_np,
                rewards,
                masks,
            )
            self.environment_steps += 1
            self.agent_decisions += self.env.n
            observations = result.observations
            if result.done:
                break

        self.trainer.prep_rollout()
        with torch.no_grad():
            next_values = self.policy.get_values(
                buffer.share_obs[-1, 0],
                buffer.rnn_states_critic[-1, 0],
                buffer.masks[-1, 0],
            ).detach().cpu().numpy()[None, ...]
        buffer.compute_returns(next_values, self.trainer.value_normalizer)
        self.trainer.prep_training()
        stats = self.trainer.train(buffer)
        buffer.after_update()
        self.updates += 1
        return {
            "episode_reward": episode_reward,
            "environment_steps": self.environment_steps,
            "agent_decisions": self.agent_decisions,
            "updates": self.updates,
            **{
                key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                for key, value in stats.items()
            },
        }

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.actor.state_dict(), directory / "actor.pt")
        torch.save(self.policy.critic.state_dict(), directory / "critic.pt")
        action_metadata = (
            "continuous latent Box(5) with joint semantic target decode"
            if self.config.action_adapter == "continuous"
            else f"MultiDiscrete(6,{self.config.control_bins},{self.config.control_bins},{self.config.control_bins},{self.env.n + 1})"
        )
        metadata = (
            f"repository=https://github.com/marlbenchmark/on-policy\n"
            f"commit={self.commit}\n"
            f"action_adapter={action_metadata}\n"
        )
        (directory / "ADAPTER_METADATA.txt").write_text(metadata, encoding="utf-8")

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        self.policy.actor.load_state_dict(torch.load(directory / "actor.pt", map_location="cpu", weights_only=True))
        self.policy.critic.load_state_dict(torch.load(directory / "critic.pt", map_location="cpu", weights_only=True))

    @property
    def source_fingerprint(self) -> str:
        license_bytes = (self.source / "LICENSE").read_bytes()
        return hashlib.sha256((self.commit.encode("ascii") + license_bytes)).hexdigest()
