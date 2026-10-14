# -*- coding: utf-8 -*-
"""P0 smoke check: env.reset -> QP shield filter -> MAPPO collect + update x2.
Recreated 2026-10-13 (original from migration session was not in package).
"""
import time

from aircraft_sim import AircraftEnv, MAPPOConfig, MAPPOPolicy, RuleMonitor, SimConfig
from aircraft_sim.qp_shield import ConflictAwareQPSafetyShield

t0 = time.time()
env = AircraftEnv(3, SimConfig(horizon=8, seed=42))
obs = env.reset()
shield = ConflictAwareQPSafetyShield(RuleMonitor(env.cfg))

# shield filter pass: joint filter over all-agent observations with rule-agnostic nominal actions
from aircraft_sim.policies import rule_agnostic_policy

interventions = 0
for _ in range(3):
    nominal = {uid: rule_agnostic_policy(ob, uid) for uid, ob in obs.items()}
    _, interv = shield.filter(obs, nominal)
    interventions += len(interv)

policy = MAPPOPolicy(env, shield, config=MAPPOConfig(), seed=42)
rollout = policy.collect(horizon=8)
stats1 = policy.update(rollout)
stats2 = policy.update(rollout)
assert stats2["loss"] == stats2["loss"], "loss NaN"
print("SMOKE_OK loss=%.3f intervention_fraction=%.3f shield_interventions=%d elapsed=%.1fs"
      % (stats2["loss"], stats2.get("intervention_fraction", -1), interventions, time.time() - t0))
