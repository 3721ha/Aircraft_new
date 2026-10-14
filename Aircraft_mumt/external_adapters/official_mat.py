"""Aircraft adapter for PKU-MARL's official Multi-Agent Transformer."""

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
from .common import Box, ContinuousAircraftActionCodec


PINNED_COMMIT = "be3ff49c8264d454c1fe2c41582aa2bfc98498c8"
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "external_methods" / "inbox" / "mat" / "SOURCE"


@dataclass
class OfficialMATConfig:
    source: Path = DEFAULT_SOURCE
    embedding_size: int = 64
    transformer_blocks: int = 1
    attention_heads: int = 1
    learning_rate: float = 5e-4
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
    package_dir = source / "mat"
    if not (package_dir / "algorithms" / "mat" / "mat_trainer.py").exists():
        raise FileNotFoundError(f"official MAT source not found: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    # The root initializer imports environment packages that are irrelevant
    # to the algorithm core and tied to MAT's obsolete benchmark stack.
    loaded = sys.modules.get("mat")
    expected_path = str(package_dir)
    if loaded is None:
        package = ModuleType("mat")
        package.__package__ = "mat"
        package.__path__ = [expected_path]
        sys.modules["mat"] = package
    elif expected_path not in [str(path) for path in getattr(loaded, "__path__", [])]:
        raise RuntimeError(f"a different mat package is already loaded: {loaded!r}")

    from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
    from mat.algorithms.mat.mat_trainer import MATTrainer
    from mat.utils.shared_buffer import SharedReplayBuffer

    return TransformerPolicy, MATTrainer, SharedReplayBuffer


class OfficialMATAdapter:
    """Run official MAT as a centralized joint-action communication baseline."""

    execution_information = "centralized_joint_local_observations"

    def __init__(self, env, config: OfficialMATConfig | None = None, seed: int = 0):
        self.env = env
        self.config = config or OfficialMATConfig()
        self.source = Path(self.config.source).resolve()
        self.commit = _commit(self.source)
        if self.config.verify_commit and self.commit != PINNED_COMMIT:
            raise RuntimeError(f"MAT source commit {self.commit} does not match pinned {PINNED_COMMIT}")
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
        self.policy = policy_type(
            self.args,
            self.obs_space,
            self.share_obs_space,
            self.action_space,
            env.n,
            torch.device("cpu"),
        )
        self.trainer = trainer_type(self.args, self.policy, env.n, torch.device("cpu"))
        self.buffer_type = buffer_type
        self.environment_steps = 0
        self.agent_decisions = 0
        self.updates = 0

    def _official_args(self):
        cfg = self.config
        return SimpleNamespace(
            algorithm_name="mat",
            env_name="Aircraft",
            lr=cfg.learning_rate,
            opti_eps=1e-5,
            weight_decay=0.0,
            use_policy_active_masks=True,
            use_value_active_masks=True,
            use_recurrent_policy=False,
            use_naive_recurrent_policy=False,
            hidden_size=cfg.embedding_size,
            recurrent_N=1,
            n_block=cfg.transformer_blocks,
            n_embd=cfg.embedding_size,
            n_head=cfg.attention_heads,
            encode_state=False,
            dec_actor=False,
            share_actor=False,
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
            use_popart=False,
            use_valuenorm=False,
            use_proper_time_limits=False,
        )

    def _encode_observations(self, observations: dict) -> np.ndarray:
        zero = [0.0] * self.encoder.dimension
        return np.asarray([
            self.encoder.encode(observations[uid], uid) if uid in observations else zero
            for uid in range(self.env.n)
        ], dtype=np.float32)

    @staticmethod
    def _shared_observations(encoded: np.ndarray) -> np.ndarray:
        shared = encoded.reshape(-1)
        return np.repeat(shared[None, :], len(encoded), axis=0).astype(np.float32)

    def decode_action(self, latent) -> Action:
        return self.action_codec.decode(latent)

    @torch.no_grad()
    def act_joint(self, observations: dict, deterministic: bool = True) -> dict[int, Action]:
        self.trainer.prep_rollout()
        encoded = self._encode_observations(observations)
        shared = self._shared_observations(encoded)
        rnn_state = np.zeros((self.env.n, 1, self.config.embedding_size), dtype=np.float32)
        mask = np.ones((self.env.n, 1), dtype=np.float32)
        latent, _ = self.policy.act(shared, encoded, rnn_state, mask, deterministic=deterministic)
        latent = latent.detach().cpu().numpy()
        return {uid: self.decode_action(latent[uid]) for uid in range(self.env.n)}

    @staticmethod
    def _scalar(value):
        if torch.is_tensor(value):
            return float(value.detach().float().mean().cpu())
        return float(np.asarray(value, dtype=float).mean())

    def train_episode(self, scenario: dict | None = None) -> dict:
        observations = self.env.reset(scenario)
        encoded = self._encode_observations(observations)
        shared = self._shared_observations(encoded)
        buffer = self.buffer_type(
            self.args, self.env.n, self.obs_space, self.share_obs_space, self.action_space, "Aircraft"
        )
        buffer.obs[0, 0] = encoded
        buffer.share_obs[0, 0] = shared
        episode_reward = 0.0

        for _ in range(self.args.episode_length):
            self.trainer.prep_rollout()
            with torch.no_grad():
                values, latent, log_prob, rnn_actor, rnn_critic = self.policy.get_actions(
                    np.concatenate(buffer.share_obs[buffer.step]),
                    np.concatenate(buffer.obs[buffer.step]),
                    np.concatenate(buffer.rnn_states[buffer.step]),
                    np.concatenate(buffer.rnn_states_critic[buffer.step]),
                    np.concatenate(buffer.masks[buffer.step]),
                )
            values_np = values.detach().cpu().numpy()[None, ...]
            latent_np = latent.detach().cpu().numpy()[None, ...]
            log_prob_np = log_prob.detach().cpu().numpy()[None, ...]
            rnn_actor_np = rnn_actor.detach().cpu().numpy()[None, ...]
            rnn_critic_np = rnn_critic.detach().cpu().numpy()[None, ...]
            action_dict = {uid: self.decode_action(latent_np[0, uid]) for uid in range(self.env.n)}
            step = self.env.step(action_dict)
            episode_reward += float(step.reward)
            next_encoded = self._encode_observations(step.observations)
            next_shared = self._shared_observations(next_encoded)
            rewards = np.full((1, self.env.n, 1), float(step.reward), dtype=np.float32)
            masks = (
                np.zeros((1, self.env.n, 1), dtype=np.float32)
                if step.done else np.ones((1, self.env.n, 1), dtype=np.float32)
            )
            buffer.insert(
                next_shared[None, ...],
                next_encoded[None, ...],
                rnn_actor_np,
                rnn_critic_np,
                latent_np,
                log_prob_np,
                values_np,
                rewards,
                masks,
            )
            observations = step.observations
            self.environment_steps += 1
            self.agent_decisions += self.env.n
            if step.done:
                break

        self.trainer.prep_rollout()
        with torch.no_grad():
            next_values = self.policy.get_values(
                np.concatenate(buffer.share_obs[-1]),
                np.concatenate(buffer.obs[-1]),
                np.concatenate(buffer.rnn_states_critic[-1]),
                np.concatenate(buffer.masks[-1]),
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
            **{key: self._scalar(value) for key, value in stats.items()},
        }

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.transformer.state_dict(), directory / "transformer.pt")
        (directory / "ADAPTER_METADATA.txt").write_text(
            "\n".join([
                "repository=https://github.com/PKU-MARL/Multi-Agent-Transformer",
                f"commit={self.commit}",
                "execution_information=centralized joint sequence of all local observations",
                "action_adapter=continuous latent Box(5) with joint semantic target decode",
                "license=not provided by upstream repository",
                "",
            ]),
            encoding="utf-8",
        )

    def load(self, directory: Path) -> None:
        state = torch.load(Path(directory) / "transformer.pt", map_location="cpu", weights_only=True)
        self.policy.transformer.load_state_dict(state)

    @property
    def source_fingerprint(self) -> str:
        return hashlib.sha256(
            self.commit.encode("ascii") + (self.source / "README.md").read_bytes()
        ).hexdigest()
