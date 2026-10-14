"""Small PyTorch MAPPO-style trainer for the decision-level simulator.

The implementation is intentionally compact: one shared actor is used by all
aircraft and the critic consumes the concatenated team observation.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import copy
from torch import nn
from torch.distributions import Categorical, Normal

from .models import Action, Mode
from .rl_api import ActionCodec, ObservationEncoder
from .rules import RuleMonitor, is_hard_rule, rule_family


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, n_agents: int, hidden: int = 128, initial_log_std: float = -1.2):
        super().__init__()
        self.n_agents = n_agents
        self.actor = nn.Sequential(nn.Linear(observation_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mode_head = nn.Linear(hidden, 6)
        self.param_head = nn.Linear(hidden, 3)
        # Zero-initialized residual head keeps a transferred actor exactly at
        # its source continuous action before JSBSim adaptation begins.
        self.residual_head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.target_head = nn.Linear(hidden, n_agents + 1)
        self.target_kind_head = nn.Linear(hidden, 4)
        self.log_std = nn.Parameter(torch.full((3,), initial_log_std))
        self.critic = nn.Sequential(nn.Linear(observation_dim * n_agents, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.cost_critic = nn.Sequential(nn.Linear(observation_dim * n_agents, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def actor_distribution(self, obs: torch.Tensor):
        h = self.actor(obs)
        return (
            Categorical(logits=self.mode_head(h)),
            Normal(self.param_head(h), self.log_std.exp()),
            Categorical(logits=self.target_head(h)),
            Categorical(logits=self.target_kind_head(h)),
        )

    def value(self, team_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(team_obs).squeeze(-1)

    def cost_value(self, team_obs: torch.Tensor) -> torch.Tensor:
        return self.cost_critic(team_obs).squeeze(-1)


@dataclass
class MAPPOConfig:
    hidden_size: int = 128
    clip_ratio: float = 0.2
    learning_rate: float = 5e-5
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    intervention_coef: float = 0.2
    epochs: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_scale: float = 0.1
    initial_log_std: float = -1.2
    target_kl: float = 0.02
    anchor_kl_coef: float = 0.0
    dwell_penalty_coef: float = 0.0
    dwell_action_coef: float = 0.0
    safety_objective: str = "task"
    fixed_penalty_coef: float = 0.35
    stl_reward_coef: float = 0.20
    cost_limit: float = 0.05
    dual_learning_rate: float = 0.05
    macpo_max_projection: float = 10.0


class MAPPOPolicy:
    def __init__(self, env, shield, config: MAPPOConfig | None = None, seed: int = 0):
        torch.manual_seed(seed)
        self.env = env
        self.shield = shield
        self.config = config or MAPPOConfig()
        self.encoder = ObservationEncoder(env.n, env.cfg.max_speed, env.cfg.max_altitude)
        self.codec = ActionCodec(env.n)
        self.model = ActorCritic(
            self.encoder.dimension,
            env.n,
            hidden=self.config.hidden_size,
            initial_log_std=self.config.initial_log_std,
        )
        self.anchor_model = None
        self.residual_scale = 0.0
        self.residual_base = None
        self.discrete_gate_enabled = False
        self.discrete_gate_threshold = 0.9
        self.discrete_gate_count = 0
        self.discrete_gate_reasons = {"resource": 0, "identification": 0, "communication": 0, "teammate_risk": 0}
        self.lagrange_multiplier = 0.0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def _training_safety_signal(self, monitor: RuleMonitor, truth: dict, actions: Dict[int, Action]) -> tuple[float, float, list[str]]:
        """Return a bounded hard-rule cost and an STL-style robustness signal.

        Full truth is used only by the simulator's training objective. It is
        never appended to observations or consumed by the decentralized actor.
        """

        report = monitor.evaluate_truth(truth, actions)
        hard_violations = sorted({rule for rule in report.violations if is_hard_rule(rule)})
        hard_cost = min(1.0, len(hard_violations) / max(1, self.env.n))
        normalized_margins = []
        scales = {
            "S01": max(1.0, self.env.cfg.max_speed - self.env.cfg.min_speed),
            "S02": max(1.0, self.env.cfg.min_separation),
            "S03": max(1.0, self.env.cfg.safe_trend_window),
            "S04": max(1e-3, self.env.cfg.min_energy),
            "F06": max(1e-3, self.env.cfg.reserve_capacity),
        }
        for name, margin in report.margins.items():
            family = rule_family(name)
            if is_hard_rule(family):
                normalized_margins.append(torch.tanh(torch.tensor(float(margin) / scales.get(family, 1.0))).item())
        if hard_violations:
            normalized_margins.append(-1.0)
        robustness = min(normalized_margins, default=1.0)
        return hard_cost, float(max(-1.0, min(1.0, robustness))), hard_violations

    def set_anchor(self) -> None:
        """Freeze the current actor as a source-policy regularization anchor."""

        self.anchor_model = copy.deepcopy(self.model).eval()
        for parameter in self.anchor_model.parameters():
            parameter.requires_grad_(False)

    def enable_residual_mode(self, scale: float = 0.2, base_action=None) -> None:
        """Restrict continuous adaptation to a bounded residual around anchor."""

        if self.anchor_model is None:
            raise RuntimeError("set_anchor() must be called before enable_residual_mode()")
        if scale <= 0.0 or scale > 1.0:
            raise ValueError("residual scale must be in (0, 1]")
        self.residual_scale = float(scale)
        if base_action is not None:
            if len(base_action) != 3:
                raise ValueError("base_action must contain turn, climb and acceleration")
            self.residual_base = tuple(float(value) for value in base_action)
        else:
            self.residual_base = None
        # A residual adapter must not rewrite the source policy's semantic
        # mode, target, or feature representation.  Only the zero-initialized
        # residual head and its scale/variance remain trainable.
        for module in (
            self.model.actor, self.model.mode_head, self.model.target_head,
            self.model.target_kind_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def enable_discrete_safety_gate(self, risk_threshold: float = 0.9) -> None:
        """Fallback only on explicit hard-risk observations."""

        if not 0.0 < risk_threshold <= 1.0:
            raise ValueError("risk_threshold must be in (0, 1]")
        self.discrete_gate_enabled = True
        self.discrete_gate_threshold = float(risk_threshold)

    def _apply_discrete_safety_gate(self, observation: dict, action: Action, agent_id: int) -> Action:
        if not self.discrete_gate_enabled:
            return action
        own = observation.get("self", {})
        teammate_risk = max(
            (float(item.get("risk", 0.0)) for item in observation.get("teammates", [])),
            default=0.0,
        )
        reasons = {
            "resource": float(own.get("resource", 1.0)) < 0.20,
            "identification": float(own.get("id_confidence", 1.0)) < 0.70,
            "communication": not bool(own.get("c2_connected", True)) and bool(own.get("warning_pending", False)),
            "teammate_risk": teammate_risk >= self.discrete_gate_threshold,
        }
        hard_trigger = any(reasons.values())
        if not hard_trigger:
            return action
        from .policies import conservative_policy

        self.discrete_gate_count += 1
        for reason, active in reasons.items():
            if active:
                self.discrete_gate_reasons[reason] += 1
        # A full conservative fallback is used for hard triggers.  This keeps
        # the residual adapter from issuing an aggressive continuous command
        # while the semantic mode is being repaired.
        if (
            float(own.get("resource", 1.0)) < 0.20
            or float(own.get("id_confidence", 1.0)) < 0.70
            or (not bool(own.get("c2_connected", True)) and bool(own.get("warning_pending", False)))
            or teammate_risk >= self.discrete_gate_threshold
        ):
            return Action(Mode.RECOVER, acceleration=-0.2)
        return conservative_policy(observation, agent_id)

    def _policy_distribution(self, obs: torch.Tensor):
        mode_dist, params, target_dist, target_kind_dist = self.model.actor_distribution(obs)
        if self.residual_scale <= 0.0:
            return mode_dist, params, target_dist, target_kind_dist
        if self.anchor_model is None:
            raise RuntimeError("residual mode requires an anchor model")
        with torch.no_grad():
            anchor_params = self.anchor_model.actor_distribution(obs)[1]
        hidden = self.model.actor(obs)
        residual_loc = self.model.residual_head(hidden)
        residual_std = self.residual_scale * self.model.log_std.exp()
        if self.residual_base is None:
            base_loc = anchor_params.loc
        else:
            base_loc = torch.tensor(self.residual_base, dtype=obs.dtype, device=obs.device).expand_as(residual_loc)
        params = Normal(base_loc + self.residual_scale * residual_loc, residual_std)
        return mode_dist, params, target_dist, target_kind_dist

    def act(self, observation: dict, agent_id: int = 0, deterministic: bool = False) -> Action:
        action, _ = self.act_with_log_prob(observation, agent_id, deterministic)
        return action

    def act_with_log_prob(self, observation: dict, agent_id: int = 0, deterministic: bool = False):
        with torch.no_grad():
            obs = torch.tensor([self.encoder.encode(observation, agent_id)], dtype=torch.float32)
            mode_dist, param_dist, target_dist, target_kind_dist = self._policy_distribution(obs)
            mode = mode_dist.probs.argmax(-1) if deterministic else mode_dist.sample()
            params = param_dist.loc if deterministic else param_dist.sample()
            target = target_dist.probs.argmax(-1) if deterministic else target_dist.sample()
            target_kind = (
                target_kind_dist.probs.argmax(-1)
                if deterministic else target_kind_dist.sample()
            )
            action = self.codec.decode([
                mode.item(), *params.squeeze(0).tolist(),
                target.item() - 1, target_kind.item(),
            ])
            action = self._apply_discrete_safety_gate(observation, action, agent_id)
            action_tensor = self._action_tensor(action).unsqueeze(0)
            log_prob, _ = self._log_prob_entropy(obs, action_tensor)
        return action, float(log_prob.item())

    def _action_tensor(self, action: Action) -> torch.Tensor:
        encoded = self.codec.encode(action)
        return torch.tensor(encoded, dtype=torch.float32)

    def _log_prob_entropy(self, obs: torch.Tensor, action: torch.Tensor):
        mode_dist, param_dist, target_dist, target_kind_dist = self._policy_distribution(obs)
        mode = action[:, 0].long().clamp(0, 5)
        params = action[:, 1:4]
        target = (action[:, 4].long() + 1).clamp(0, self.env.n)
        target_kind = action[:, 5].long().clamp(0, 3)
        teammate_mask = (target_kind == 0).to(dtype=params.dtype)
        logp = (
            mode_dist.log_prob(mode)
            + param_dist.log_prob(params).sum(-1)
            + target_dist.log_prob(target) * teammate_mask
            + target_kind_dist.log_prob(target_kind)
        )
        entropy = (
            mode_dist.entropy()
            + param_dist.entropy().sum(-1)
            + target_dist.entropy() * teammate_mask
            + target_kind_dist.entropy()
        )
        return logp, entropy

    def behavior_clone(self, demonstrations: Sequence[tuple], epochs: int = 100, learning_rate: float = 1e-3) -> dict:
        """Warm-start the actor from observation/action demonstrations.

        Demonstrations contain only policy-visible observations and nominal
        actions. No truth state or rule-monitor output is consumed here.
        """

        if not demonstrations:
            return {"behavior_clone_loss": 0.0, "samples": 0}
        observations = torch.tensor(
            [self.encoder.encode(observation, agent_id) for observation, agent_id, _ in demonstrations],
            dtype=torch.float32,
        )
        actions = torch.stack([self._action_tensor(action) for _, _, action in demonstrations])
        actor_optimizer = torch.optim.Adam(
            list(self.model.actor.parameters())
            + list(self.model.mode_head.parameters())
            + list(self.model.param_head.parameters())
            + list(self.model.target_head.parameters())
            + list(self.model.target_kind_head.parameters()),
            lr=learning_rate,
        )
        loss = torch.zeros(())
        for _ in range(epochs):
            hidden = self.model.actor(observations)
            mode_loss = nn.functional.cross_entropy(self.model.mode_head(hidden), actions[:, 0].long().clamp(0, 5))
            parameter_loss = ((self.model.param_head(hidden) - actions[:, 1:4]) ** 2).mean()
            target_loss = nn.functional.cross_entropy(self.model.target_head(hidden), (actions[:, 4].long() + 1).clamp(0, self.env.n))
            target_kind_loss = nn.functional.cross_entropy(
                self.model.target_kind_head(hidden), actions[:, 5].long().clamp(0, 3)
            )
            loss = mode_loss + parameter_loss + target_loss + target_kind_loss
            actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            actor_optimizer.step()
        with torch.no_grad():
            hidden = self.model.actor(observations)
            mode_accuracy = (self.model.mode_head(hidden).argmax(-1) == actions[:, 0].long()).float().mean()
        return {"behavior_clone_loss": float(loss.detach()), "mode_accuracy": float(mode_accuracy), "samples": len(demonstrations)}

    def collect(self, scenario: dict | None = None, horizon: int | None = None, deterministic: bool = False) -> List[dict]:
        observations = self.env.reset(scenario)
        if hasattr(self.shield, "monitor") and hasattr(self.shield.monitor, "reset"):
            self.shield.monitor.reset()
        training_monitor = RuleMonitor(self.env.cfg)
        data = []
        for _ in range(horizon or self.env.cfg.horizon):
            sampled = {i: self.act_with_log_prob(observations[i], i, deterministic) for i in observations}
            nominal = {i: value[0] for i, value in sampled.items()}
            old_log_prob = {i: value[1] for i, value in sampled.items()}
            safe, interventions = self.shield.filter(observations, nominal)
            result = self.env.step(safe)
            encoded = {i: self.encoder.encode(observations[i], i) for i in observations}
            zero = [0.0] * self.encoder.dimension
            team_observation = sum((encoded.get(i, zero) for i in range(self.env.n)), [])
            with torch.no_grad():
                team_tensor = torch.tensor([team_observation], dtype=torch.float32)
                value = float(self.model.value(team_tensor).item())
                cost_value = float(self.model.cost_value(team_tensor).item())
            role_changes = {
                i: int(
                    0 < int(observations[i].get("self", {}).get("mode_age", 0)) < self.env.cfg.mode_dwell_steps
                    and observations[i].get("self", {}).get("role") != nominal[i].mode.value
                )
                for i in nominal
            }
            safety_cost, stl_robustness, hard_violations = self._training_safety_signal(
                training_monitor, result.info["truth"], safe
            )
            item = {"observations": encoded, "team_observation": team_observation, "nominal": nominal, "safe": safe, "old_log_prob": old_log_prob, "value": value, "cost_value": cost_value, "reward": result.reward, "safety_cost": safety_cost, "stl_robustness": stl_robustness, "training_hard_violations": hard_violations, "done": result.done, "interventions": interventions, "role_changes": role_changes}
            if hasattr(self.shield, "last_solution"):
                item["belief_report"] = self.shield.last_solution.belief_report
                item["qp_solution"] = self.shield.last_solution
            data.append(item)
            observations = result.observations
            if result.done:
                break
        return data

    def update(self, rollout: Sequence[dict]) -> dict:
        if not rollout:
            return {"loss": 0.0, "policy_loss": 0.0, "intervention_loss": 0.0}
        obs, team, actions, safe_actions, old_log_probs = [], [], [], [], []
        rewards, costs, values, cost_values, dones, role_changes = [], [], [], [], [], []
        objective = self.config.safety_objective
        if objective not in {"task", "fixed_penalty", "stl_reward", "lagrangian", "macpo"}:
            raise ValueError(f"unknown safety objective: {objective}")
        mean_rollout_cost = sum(float(item.get("safety_cost", 0.0)) for item in rollout) / len(rollout)
        if objective == "lagrangian":
            self.lagrange_multiplier = max(
                0.0,
                self.lagrange_multiplier
                + self.config.dual_learning_rate * (mean_rollout_cost - self.config.cost_limit),
            )
        for item in rollout:
            for i in range(self.env.n):
                obs.append(item["observations"].get(i, [0.0] * self.encoder.dimension))
                actions.append(self._action_tensor(item["nominal"].get(i, Action())))
                safe_actions.append(self._action_tensor(item["safe"].get(i, Action())))
                old_log_probs.append(float(item.get("old_log_prob", {}).get(i, 0.0)))
            team.append(item["team_observation"])
            training_reward = (
                float(item["reward"]) * self.config.reward_scale
                - self.config.dwell_penalty_coef * sum(item.get("role_changes", {}).values()) / max(1, self.env.n)
            )
            safety_cost = float(item.get("safety_cost", 0.0))
            if objective == "fixed_penalty":
                training_reward -= self.config.fixed_penalty_coef * safety_cost
            elif objective == "stl_reward":
                training_reward += self.config.stl_reward_coef * float(item.get("stl_robustness", 0.0))
            elif objective == "lagrangian":
                training_reward -= self.lagrange_multiplier * safety_cost
            rewards.append(training_reward)
            costs.append(safety_cost)
            values.append(float(item.get("value", 0.0)))
            cost_values.append(float(item.get("cost_value", 0.0)))
            dones.append(bool(item.get("done", False)))
            role_changes.append(sum(item.get("role_changes", {}).values()))
        obs_t = torch.tensor(obs, dtype=torch.float32)
        team_t = torch.tensor(team, dtype=torch.float32)
        act_t = torch.stack(actions)
        safe_t = torch.stack(safe_actions)
        # GAE preserves the temporal credit assignment required by MAPPO.
        # Episode boundaries in a concatenated rollout reset the recursion.
        advantages_by_step = [0.0] * len(rollout)
        gae = 0.0
        next_value = 0.0
        for index in range(len(rollout) - 1, -1, -1):
            nonterminal = 0.0 if dones[index] else 1.0
            delta = rewards[index] + self.config.gamma * next_value * nonterminal - values[index]
            gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * gae
            advantages_by_step[index] = gae
            next_value = values[index]
        returns_by_step = torch.tensor([advantages_by_step[i] + values[i] for i in range(len(rollout))], dtype=torch.float32)
        advantages = torch.tensor(advantages_by_step, dtype=torch.float32).repeat_interleave(self.env.n)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cost_advantages_by_step = [0.0] * len(rollout)
        cost_gae = 0.0
        next_cost_value = 0.0
        for index in range(len(rollout) - 1, -1, -1):
            nonterminal = 0.0 if dones[index] else 1.0
            delta = costs[index] + self.config.gamma * next_cost_value * nonterminal - cost_values[index]
            cost_gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * cost_gae
            cost_advantages_by_step[index] = cost_gae
            next_cost_value = cost_values[index]
        cost_returns_by_step = torch.tensor(
            [cost_advantages_by_step[i] + cost_values[i] for i in range(len(rollout))],
            dtype=torch.float32,
        )
        cost_advantages = torch.tensor(cost_advantages_by_step, dtype=torch.float32).repeat_interleave(self.env.n)
        old_logp = torch.tensor(old_log_probs, dtype=torch.float32)
        intervention_mask = (act_t - safe_t).abs().sum(-1) > 1e-8
        stats = {}
        dwell_penalty = self.config.dwell_penalty_coef * sum(role_changes) / max(1, len(rollout) * self.env.n)
        approx_kl = torch.zeros(())
        dwell_action_loss = torch.zeros(())
        cost_value_loss = torch.zeros(())
        cost_surrogate = torch.zeros(())
        projection_multiplier = 0.0
        constraint_violation = mean_rollout_cost - self.config.cost_limit
        for _ in range(self.config.epochs):
            logp, entropy = self._log_prob_entropy(obs_t, act_t)
            log_ratio = (logp - old_logp).clamp(-10.0, 10.0)
            ratio = log_ratio.exp()
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clipped = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
            policy_loss = -torch.min(ratio * advantages, clipped).mean()
            value_loss = ((self.model.value(team_t) - returns_by_step) ** 2).mean()
            if objective == "macpo":
                cost_surrogate = (ratio * cost_advantages).mean()
                cost_value_loss = ((self.model.cost_value(team_t) - cost_returns_by_step) ** 2).mean()
                if constraint_violation > 0.0:
                    actor_parameters = [
                        parameter
                        for module in (
                            self.model.actor, self.model.mode_head, self.model.param_head,
                            self.model.target_head, self.model.target_kind_head,
                        )
                        for parameter in module.parameters()
                        if parameter.requires_grad
                    ] + [self.model.log_std]
                    reward_grad = torch.autograd.grad(policy_loss, actor_parameters, retain_graph=True, allow_unused=True)
                    cost_grad = torch.autograd.grad(cost_surrogate, actor_parameters, retain_graph=True, allow_unused=True)
                    dot_gc = sum(
                        float((g * c).sum().detach())
                        for g, c in zip(reward_grad, cost_grad)
                        if g is not None and c is not None
                    )
                    norm_c = sum(
                        float((c * c).sum().detach()) for c in cost_grad if c is not None
                    )
                    projection_multiplier = min(
                        self.config.macpo_max_projection,
                        max(0.0, (constraint_violation - dot_gc) / (norm_c + 1e-8)),
                    )
                    policy_loss = policy_loss + projection_multiplier * cost_surrogate
            if self.config.dwell_action_coef > 0.0:
                # ObservationEncoder stores role at index 15 and normalized
                # mode age at index 17. While the dwell window is active,
                # softly train the mode head to retain the current role.
                role_index = (obs_t[:, 15] * (len(Mode) - 1)).round().long().clamp(0, len(Mode) - 1)
                mode_age = obs_t[:, 17] * getattr(self.encoder, "_dwell_norm", 10.0)
                dwell_mask = (mode_age > 0.0) & (mode_age < float(self.env.cfg.mode_dwell_steps))
                if dwell_mask.any():
                    dwell_logits = self.model.mode_head(self.model.actor(obs_t[dwell_mask]))
                    dwell_action_loss = nn.functional.cross_entropy(dwell_logits, role_index[dwell_mask])
                else:
                    dwell_action_loss = torch.zeros((), device=obs_t.device)
            # Intervention-consistent learning only imitates actual shield
            # corrections. Including untouched stochastic actions here turns
            # distillation into self-imitation noise and destabilizes MAPPO.
            if intervention_mask.any():
                masked_obs = obs_t[intervention_mask]
                masked_safe = safe_t[intervention_mask]
                hidden = self.model.actor(masked_obs)
                safe_mode_loss = nn.functional.cross_entropy(self.model.mode_head(hidden), masked_safe[:, 0].long().clamp(0, 5))
                safe_param_loss = ((self.model.param_head(hidden) - masked_safe[:, 1:4]) ** 2).mean()
                safe_target_loss = nn.functional.cross_entropy(self.model.target_head(hidden), (masked_safe[:, 4].long() + 1).clamp(0, self.env.n))
                safe_target_kind_loss = nn.functional.cross_entropy(
                    self.model.target_kind_head(hidden),
                    masked_safe[:, 5].long().clamp(0, 3),
                )
                intervention_loss = (
                    safe_mode_loss + safe_param_loss + safe_target_loss
                    + safe_target_kind_loss
                )
            else:
                intervention_loss = torch.zeros((), dtype=torch.float32)
            anchor_kl = torch.zeros((), dtype=torch.float32)
            if self.anchor_model is not None and self.config.anchor_kl_coef > 0.0:
                with torch.no_grad():
                    anchor_mode, anchor_params, anchor_target, anchor_target_kind = self.anchor_model.actor_distribution(obs_t)
                current_mode, current_params, current_target, current_target_kind = self._policy_distribution(obs_t)
                anchor_kl = torch.distributions.kl_divergence(anchor_mode, current_mode).mean()
                anchor_kl = anchor_kl + torch.distributions.kl_divergence(anchor_target, current_target).mean()
                anchor_kl = anchor_kl + torch.distributions.kl_divergence(
                    anchor_target_kind, current_target_kind
                ).mean()
                if self.residual_base is None:
                    anchor_kl = anchor_kl + torch.distributions.kl_divergence(anchor_params, current_params).sum(-1).mean()
            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean() + self.config.intervention_coef * intervention_loss + self.config.anchor_kl_coef * anchor_kl + self.config.dwell_action_coef * dwell_action_loss
            if objective == "macpo":
                loss = loss + self.config.value_coef * cost_value_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            with torch.no_grad():
                self.model.log_std.clamp_(-2.5, 0.0)
            if float(approx_kl.detach()) > self.config.target_kl:
                break
        stats.update({"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()), "cost_value_loss": float(cost_value_loss.detach()), "cost_surrogate": float(cost_surrogate.detach()), "constraint_violation": float(constraint_violation), "projection_multiplier": float(projection_multiplier), "lagrange_multiplier": float(self.lagrange_multiplier), "mean_safety_cost": float(mean_rollout_cost), "safety_objective": objective, "intervention_loss": float(intervention_loss.detach()), "anchor_kl": float(anchor_kl.detach()), "dwell_action_loss": float(dwell_action_loss.detach()), "mean_return": float(returns_by_step.mean()), "intervention_fraction": float(intervention_mask.float().mean()), "approx_kl": float(approx_kl.detach()), "dwell_penalty": float(dwell_penalty), "role_change_rate": float(sum(role_changes) / max(1, len(rollout) * self.env.n)), "rollout_steps": len(rollout), "agent_decisions": len(rollout) * self.env.n})
        return stats
