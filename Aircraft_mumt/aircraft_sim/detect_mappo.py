"""MAPPO trainer for the MUMT detection scenario (hybrid action space).

Hybrid heads (plan §7.2): 3 continuous (u_psi, u_v, u_xi) + role (2) + gear
(5) + pointer (9).  One shared actor for all 3 wingmen; critic consumes the
concatenated team observation (centralised training, decentralised
execution).  Gear logits are masked by the sampled role at both collection
and update time (SEARCH -> {45, 60}, TRACK -> {7.5, 15, 30}, plan §6.1
role-gear mask), so PPO ratios are computed on the masked distribution the
env actually sees.

Intervention-consistent learning (一行不动, safety-chain component 6): PPO
ratios use the *nominal* actions; on steps where the shield corrected an
agent, an auxiliary imitation loss pulls the nominal heads toward the safe
action (coef = intervention_coef), so the policy learns to stop provoking
interventions instead of relying on them.
"""
from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .detect_config import DetectSimConfig
from .detect_encoder import DetectObservationEncoder, DetectActionCodec, GEAR_ORDER
from .detect_models import DetectAction, DetectRole

ACT_DIM = 6        # [u_psi, u_v, u_xi, role, gear, pointer]


def _gear_masked_logits(gear_logits: torch.Tensor, role_idx: torch.Tensor,
                        allow_idx: torch.Tensor | None = None) -> torch.Tensor:
    """Role-gear mask (plan §6.1).  ``allow_idx`` (teacher/safe target gear) is
    unioned into the legal set so a lagging radar switch (teacher gear outside
    the role's legal set) never puts -inf on the target logit."""
    mask = torch.full_like(gear_logits, -1e9)
    track = role_idx == 1
    mask[~track, 3:] = 0.0      # single-expression indexing: in-place
    mask[track, :3] = 0.0       # (mask[~track][:, 3:] writes to a copy!)
    if allow_idx is not None:
        mask[torch.arange(gear_logits.shape[0]), allow_idx.long().clamp(0, 4)] = 0.0
    return gear_logits + mask


class DetectActorCritic(nn.Module):
    def __init__(self, observation_dim: int = 138, n_agents: int = 3,
                 hidden: int = 128, initial_log_std: float = -1.2):
        super().__init__()
        self.n_agents = n_agents
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.cont_head = nn.Linear(hidden, 3)
        self.role_head = nn.Linear(hidden, 2)
        self.gear_head = nn.Linear(hidden, 5)
        self.pointer_head = nn.Linear(hidden, 9)
        self.log_std = nn.Parameter(torch.full((3,), initial_log_std))
        self.critic = nn.Sequential(
            nn.Linear(observation_dim * n_agents, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def actor_distribution(self, obs: torch.Tensor, role_idx: torch.Tensor | None = None):
        h = self.actor(obs)
        cont = Normal(self.cont_head(h), self.log_std.exp())
        role = Categorical(logits=self.role_head(h))
        gear_logits = self.gear_head(h)
        if role_idx is not None:
            # SEARCH(0) -> gears {45,60} = idx {3,4}; TRACK(1) -> idx {0,1,2}
            mask = torch.full_like(gear_logits, -1e9)
            track = role_idx == 1
            mask[~track, 3:] = 0.0
            mask[track, :3] = 0.0
            gear_logits = gear_logits + mask
        gear = Categorical(logits=gear_logits)
        pointer = Categorical(logits=self.pointer_head(h))
        return cont, role, gear, pointer

    def value(self, team_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(team_obs).squeeze(-1)


@dataclass
class DetectMAPPOConfig:
    hidden_size: int = 128
    clip_ratio: float = 0.2
    learning_rate: float = 1e-4        # gate-5 conservative (3e-4 -> 1e-4: K=2 segment degraded policy)
    bc_learning_rate: float = 1e-3
    value_coef: float = 0.5
    entropy_coef: float = 0.01         # gate-6 exploration bump (0.001 -> 0.01: loitering plateau, KL pinned ~0.002)
    intervention_coef: float = 0.2
    epochs: int = 2                    # gate-5 conservative (4 -> 2)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_scale: float = 0.1
    initial_log_std: float = -1.2
    target_kl: float = 0.02
    # -- gate-6 potential-based shaping (TRAINING ONLY; eval untouched) ------
    # F = gamma*Phi(s') - Phi(s), Phi = -kappa * sum over owned tracks of
    # min(age, cap)/cap.  Bridges the 30 s credit chain to w1/w3 (Ng 1999:
    # potential-based F provably preserves the optimal policy).
    shaping_kappa: float = 0.05
    shaping_age_cap: float = 8.0


class DetectMAPPOPolicy:
    def __init__(self, env, shield, config: DetectMAPPOConfig | None = None,
                 seed: int = 0):
        torch.manual_seed(seed)
        self.env = env
        self.shield = shield
        self.config = config or DetectMAPPOConfig()
        self.encoder = DetectObservationEncoder(env.cfg)
        self.codec = DetectActionCodec()
        self.model = DetectActorCritic(
            self.encoder.dimension, env.cfg.N_WINGMEN,
            hidden=self.config.hidden_size,
            initial_log_std=self.config.initial_log_std)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.learning_rate)

    # ------------------------------------------------------------------
    @staticmethod
    def encode_action(a: DetectAction) -> List[float]:
        heads = DetectActionCodec().heads(a)
        return [a.u_psi, a.u_v, a.u_xi,
                float(heads["role"]), float(heads["gear"]), float(heads["pointer"])]

    def _log_prob_entropy(self, obs: torch.Tensor, act: torch.Tensor):
        role_idx = act[:, 3].long().clamp(0, 1)
        cont, role, gear, pointer = self.model.actor_distribution(obs, role_idx)
        cont_a = act[:, :3].clamp(-1.0, 1.0)
        logp = (cont.log_prob(cont_a).sum(-1)
                + role.log_prob(role_idx)
                + gear.log_prob(act[:, 4].long().clamp(0, 4))
                + pointer.log_prob(act[:, 5].long().clamp(0, 8)))
        entropy = (cont.entropy().sum(-1) + role.entropy()
                   + gear.entropy() + pointer.entropy())
        return logp, entropy

    def act_with_log_prob(self, observation: dict, deterministic: bool = False):
        actions, logps = self.act_batch({0: observation}, deterministic)
        return actions[0], logps[0]

    def act_batch(self, observations: Dict[int, dict], deterministic: bool = False):
        """One batched forward for all agents (collection hot path)."""
        ids = sorted(observations)
        obs_t = torch.tensor(
            np.asarray([self.encoder.encode(observations[i]) for i in ids]),
            dtype=torch.float32)
        with torch.no_grad():
            h = self.model.actor(obs_t)
            cont_loc = self.model.cont_head(h)
            cont_dist = Normal(cont_loc, self.model.log_std.exp())
            role_logits = self.model.role_head(h)
            pointer_logits = self.model.pointer_head(h)
            if deterministic:
                cont = cont_loc.clamp(-1.0, 1.0)
                role_idx = role_logits.argmax(-1)
                gear_logits = _gear_masked_logits(self.model.gear_head(h), role_idx)
                gear_idx = gear_logits.argmax(-1)
                pointer_idx = pointer_logits.argmax(-1)
            else:
                cont = cont_dist.sample().clamp(-1.0, 1.0)
                role_idx = Categorical(logits=role_logits).sample()
                gear_logits = _gear_masked_logits(self.model.gear_head(h), role_idx)
                gear_idx = Categorical(logits=gear_logits).sample()
                pointer_idx = Categorical(logits=pointer_logits).sample()
            logp = (cont_dist.log_prob(cont).sum(-1)
                    + Categorical(logits=role_logits).log_prob(role_idx)
                    + Categorical(logits=gear_logits).log_prob(gear_idx)
                    + Categorical(logits=pointer_logits).log_prob(pointer_idx))
        actions, logps = {}, {}
        for r, i in enumerate(ids):
            actions[i] = self.codec.to_action(cont[r].tolist(),
                                              int(role_idx[r]), int(gear_idx[r]),
                                              int(pointer_idx[r]))
            logps[i] = float(logp[r])
        return actions, logps

    # ------------------------------------------------------------------
    def collect(self, k_red: int = 4, seed: int = 0,
                deterministic: bool = False,
                w8_scale: float | None = None) -> List[dict]:
        cfg: DetectSimConfig = self.env.cfg
        scenario = {"seed": seed, "k_red": k_red}
        if w8_scale is not None:            # gate-6 penalty curriculum (train only)
            scenario["w8_scale"] = w8_scale
        obs = self.env.reset(scenario)
        data: List[dict] = []
        while True:
            encoded = {i: self.encoder.encode(obs[i]) for i in obs}
            nominal, old_logp = self.act_batch(obs, deterministic)
            safe, interventions = self.shield.filter(self.env, nominal)
            phi_before = self._track_potential()
            res = self.env.step(safe)
            team = np.concatenate([
                encoded.get(i, np.zeros(self.encoder.dimension, dtype=np.float32))
                for i in range(cfg.N_WINGMEN)]).astype(np.float32)
            with torch.no_grad():
                value = float(self.model.value(
                    torch.tensor([team], dtype=torch.float32)).item())
            # potential-based shaping: terminal states grounded at Phi = 0
            phi_after = 0.0 if res.done else self._track_potential()
            shaped = float(res.reward) * self.config.reward_scale \
                + self.config.gamma * phi_after - phi_before
            data.append({
                "obs": encoded, "team": team,
                "nominal": {i: self.encode_action(nominal[i]) for i in nominal},
                "safe": {i: self.encode_action(safe[i]) for i in safe},
                "old_logp": old_logp, "value": value,
                "reward": shaped,
                "done": res.done,
                "intervened": {it.uid: it.kind for it in interventions},
                "n_interventions": len(interventions),
                "n_qp": sum(1 for it in interventions if it.kind == "continuous_qp"),
                "n_fallback": sum(1 for it in interventions if it.kind == "reference_fallback"),
                "violations": len(res.info["report"].violations),
                "terminal": res.info["terminal"],
            })
            obs = res.observations
            if res.done:
                break
        return data

    # ------------------------------------------------------------------
    def _track_potential(self) -> float:
        """Gate-6 shaping potential: -kappa * freshness deficit of OWNED tracks.

        Team-level (matches the team reward). Only targets this wingman owns
        (responsibility == i) count, mirroring w1's own-responsibility scope."""
        cfg = self.config
        if cfg.shaping_kappa <= 0.0:
            return 0.0
        env = self.env
        deficit = 0.0
        for i, w in enumerate(env.wingmen):
            for tr in w.tracks.slots:
                if tr is None:
                    continue
                if env.responsibility.get(tr.target_id) != i:
                    continue
                deficit += min(tr.age(env.t), cfg.shaping_age_cap) / cfg.shaping_age_cap
        return -cfg.shaping_kappa * deficit

    # ------------------------------------------------------------------
    def update(self, rollout: Sequence[dict]) -> dict:
        if not rollout:
            return {"loss": 0.0}
        cfg = self.config
        n = self.env.cfg.N_WINGMEN
        obs, team, acts, safes, old_logp = [], [], [], [], []
        rewards, values, dones = [], [], []
        for item in rollout:
            for i in range(n):
                obs.append(item["obs"][i])
                acts.append(item["nominal"][i])
                safes.append(item["safe"][i])
                old_logp.append(item["old_logp"][i])
            team.append(item["team"])
            rewards.append(item["reward"])
            values.append(item["value"])
            dones.append(item["done"])
        obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32)
        team_t = torch.tensor(np.asarray(team), dtype=torch.float32)
        act_t = torch.tensor(acts, dtype=torch.float32)
        safe_t = torch.tensor(safes, dtype=torch.float32)
        old_logp_t = torch.tensor(old_logp, dtype=torch.float32)

        # GAE over episode-boundary-aware concatenated rollout
        adv = [0.0] * len(rollout)
        gae = 0.0
        next_value = 0.0
        for t in range(len(rollout) - 1, -1, -1):
            nonterminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + cfg.gamma * next_value * nonterminal - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            adv[t] = gae
            next_value = values[t]
        returns = torch.tensor([adv[t] + values[t] for t in range(len(rollout))],
                               dtype=torch.float32)
        adv_t = torch.tensor(adv, dtype=torch.float32).repeat_interleave(n)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        intervention_mask = (act_t - safe_t).abs().sum(-1) > 1e-8
        stats = {}
        approx_kl = torch.zeros(())
        for _ in range(cfg.epochs):
            logp, entropy = self._log_prob_entropy(obs_t, act_t)
            log_ratio = (logp - old_logp_t).clamp(-10.0, 10.0)
            ratio = log_ratio.exp()
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clipped = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_t
            policy_loss = -torch.min(ratio * adv_t, clipped).mean()
            value_loss_raw = ((self.model.value(team_t) - returns) ** 2).mean()
            # relative value loss (pilot calibration): raw squared error on
            # ~-200-scale returns floods the shared grad-norm clip and pins
            # the actor's KL at ~0.  Normalising by target variance keeps
            # critic learning O(1) without touching reward design.
            value_loss = value_loss_raw / (returns.var().detach() + 1e-4)
            if intervention_mask.any():
                m_obs = obs_t[intervention_mask]
                m_safe = safe_t[intervention_mask]
                h = self.model.actor(m_obs)
                s_role = m_safe[:, 3].long().clamp(0, 1)
                cont_loss = ((self.model.cont_head(h) - m_safe[:, :3].clamp(-1, 1)) ** 2).mean()
                role_loss = nn.functional.cross_entropy(self.model.role_head(h), s_role)
                gear_loss = nn.functional.cross_entropy(
                    _gear_masked_logits(self.model.gear_head(h), s_role,
                                        allow_idx=m_safe[:, 4]),
                    m_safe[:, 4].long().clamp(0, 4))
                pointer_loss = nn.functional.cross_entropy(
                    self.model.pointer_head(h), m_safe[:, 5].long().clamp(0, 8))
                intervention_loss = cont_loss + role_loss + gear_loss + pointer_loss
            else:
                intervention_loss = torch.zeros(())
            loss = (policy_loss + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy.mean()
                    + cfg.intervention_coef * intervention_loss)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            with torch.no_grad():
                self.model.log_std.clamp_(-2.5, 0.0)
            if float(approx_kl.detach()) > cfg.target_kl:
                break
        stats.update({
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "intervention_loss": float(intervention_loss.detach()),
            "intervention_fraction": float(intervention_mask.float().mean()),
            "approx_kl": float(approx_kl.detach()),
            "mean_return_scaled": float(returns.mean()),
            "rollout_steps": len(rollout),
        })
        return stats

    # ------------------------------------------------------------------
    def behavior_clone(self, demos: Sequence[Tuple[np.ndarray, DetectAction]],
                       epochs: int = 60, batch_size: int = 4096,
                       focus_weight: float = 5.0) -> dict:
        """BC warm start on HEUR demonstrations (plan §9.4, 50-100 epochs).

        ``focus_weight`` up-weights the rare-but-critical responsibility
        request steps (pointer > 0, ~23% of teacher steps): plain CE under
        class imbalance collapses the argmax to "never request", which
        starves the whole task loop (no ownership -> no TRACK regime)."""
        obs = torch.tensor(np.asarray([d[0] for d in demos]), dtype=torch.float32)
        act = torch.tensor([self.encode_action(d[1]) for d in demos],
                           dtype=torch.float32)
        samp_w = torch.where(act[:, 5] > 0,
                             torch.tensor(focus_weight), torch.tensor(1.0))
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config.bc_learning_rate)
        n = obs.shape[0]
        last = {}
        for epoch in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, batch_size):
                b = perm[s:s + batch_size]
                ob, ac, sw = obs[b], act[b], samp_w[b]
                role_idx = ac[:, 3].long().clamp(0, 1)
                h = self.model.actor(ob)
                cont_loss = (((self.model.cont_head(h) - ac[:, :3].clamp(-1, 1)) ** 2)
                             .mean(-1) * sw).mean()
                ce = nn.functional.cross_entropy
                role_loss = (ce(self.model.role_head(h), role_idx,
                                reduction="none") * sw).mean()
                gear_loss = (ce(_gear_masked_logits(self.model.gear_head(h), role_idx,
                                                    allow_idx=ac[:, 4]),
                                ac[:, 4].long().clamp(0, 4),
                                reduction="none") * sw).mean()
                pointer_loss = (ce(self.model.pointer_head(h),
                                   ac[:, 5].long().clamp(0, 8),
                                   reduction="none") * sw).mean()
                loss = cont_loss + role_loss + gear_loss + pointer_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
            last = {"epoch": epoch + 1, "loss": float(loss.detach()),
                    "cont": float(cont_loss.detach()),
                    "role": float(role_loss.detach()),
                    "gear": float(gear_loss.detach()),
                    "pointer": float(pointer_loss.detach())}
        # freeze the std closer to the demonstrated (deterministic) behaviour
        # (pilot calibration: cap -0.7 -> -1.4; std 0.5 swamped tracking loops)
        with torch.no_grad():
            self.model.log_std.clamp_(-2.5, -1.4)
        return {"samples": n, **last}

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(),
                    "config": self.config.__dict__}, path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location="cpu")["model"])
