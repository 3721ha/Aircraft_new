import unittest

from aircraft_sim import Action, AircraftEnv, ConflictAwareQPSafetyShield, Mode, RuleAgentDependencyGraph, RuleMonitor, SimConfig
from aircraft_sim.scenarios import (capability_dwell_conflict_scenario, communication_intent_conflict_scenario,
                                    coverage_resource_conflict_scenario, dual_support_conflict_scenario,
                                    protocol_conflict_scenario, risk_continuity_conflict_scenario,
                                    support_separation_conflict_scenario)


class RuleAgentGraphTests(unittest.TestCase):
    def test_graph_maps_local_pair_and_global_rules(self):
        graph = RuleAgentDependencyGraph().build(
            ["I02:0", "S02:0-1", "C02"],
            {"I02:0": 0.8, "S02:0-1": 0.6, "C02": 1.0},
            {0: Action(Mode.CRITICAL), 1: Action(Mode.HOLD), 2: Action(Mode.HOLD)},
        )
        self.assertEqual(graph.edges["I02:0"], (0,))
        self.assertEqual(graph.edges["S02:0-1"], (0, 1))
        self.assertEqual(graph.edges["C02"], (0, 1, 2))

    def test_graph_detects_cross_agent_rule_conflict(self):
        graph = RuleAgentDependencyGraph().build(
            ["C02", "F04:0"],
            {"C02": 1.0, "F04:0": 1.0},
            {0: Action(Mode.COVER), 1: Action(Mode.HOLD)},
        )
        conflicts = [event for event in graph.conflicts if event.kind == "rule_rule"]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(set(conflicts[0].involved_rules), {"C02", "F04:0"})
        self.assertGreater(graph.rules["F04:0"].dynamic_priority, graph.rules["C02"].dynamic_priority)

    def test_shield_exports_per_agent_conflict_trace(self):
        env = AircraftEnv(3, SimConfig(support_response_deadline=0))
        observations = env.reset(protocol_conflict_scenario(3))
        shield = ConflictAwareQPSafetyShield(env.cfg)
        shield.filter(observations, {i: Action(Mode.CRITICAL) for i in observations})
        solution = shield.last_solution
        self.assertIsNotNone(solution.dependency_graph)
        self.assertIn(0, solution.agent_rule_conflicts)
        self.assertTrue(any("I04:0" == rule for rule in solution.agent_rule_conflicts[0]))
        self.assertTrue(solution.conflict_events)
        self.assertTrue(all("involved_agents" in event for event in solution.conflict_events))

    def test_physical_resource_rule_beats_coverage_rule(self):
        env = AircraftEnv(3, SimConfig(noise_position=0.0, noise_speed=0.0))
        observations = env.reset(coverage_resource_conflict_scenario(3))
        shield = ConflictAwareQPSafetyShield(env.cfg)
        safe, _ = shield.filter(observations, {0: Action(Mode.COVER), 1: Action(Mode.HOLD), 2: Action(Mode.HOLD)})
        self.assertIn(safe[0].mode, {Mode.EXIT, Mode.RECOVER})
        self.assertTrue(any(action.mode == Mode.COVER for uid, action in safe.items() if uid != 0))
        pairs = [set(event["involved_rules"]) for event in shield.last_solution.conflict_events if event["kind"] == "rule_rule"]
        self.assertIn({"C02", "F04:0"}, pairs)

    def test_support_deadline_from_step_zero_is_observable(self):
        env = AircraftEnv(3, SimConfig(support_response_deadline=0))
        env.reset({
            "positions": [(0, 0, 5000), (0, 1800, 5000), (0, 3600, 5000)],
            "resources": [0.8, 0.8, 0.8],
            "support_requested": {0: True},
        })
        result = env.step({i: Action(Mode.HOLD) for i in range(3)})
        report = RuleMonitor(env.cfg).evaluate_truth(result.info["truth"], result.info["actions"])
        self.assertIn("C03:0", report.violations)

    def test_support_separation_template_exposes_the_intended_rule_pair(self):
        cfg = SimConfig(noise_position=0.0, noise_speed=0.0, support_response_deadline=0, belief_ttc_threshold=5.0, safe_trend_window=4)
        env = AircraftEnv(3, cfg)
        observations = env.reset(support_separation_conflict_scenario(3))
        shield = ConflictAwareQPSafetyShield(cfg)
        nominal = {
            0: Action(Mode.HOLD, turn=0.08, acceleration=0.1),
            1: Action(Mode.COVER, turn=0.08, acceleration=0.1),
            2: Action(Mode.HOLD, turn=0.08, acceleration=0.1),
        }
        shield.filter(observations, nominal)
        pairs = [set(event["involved_rules"]) for event in shield.last_solution.conflict_events if event["kind"] == "rule_rule"]
        self.assertIn({"C03:0", "S03:0-1"}, pairs)

    def test_dual_support_template_records_shared_helper_conflict(self):
        cfg = SimConfig(noise_position=0.0, noise_speed=0.0)
        env = AircraftEnv(3, cfg)
        observations = env.reset(dual_support_conflict_scenario(3))
        shield = ConflictAwareQPSafetyShield(cfg)
        shield.filter(observations, {
            0: Action(Mode.HOLD), 1: Action(Mode.HOLD), 2: Action(Mode.COVER),
        })
        pairs = [set(event["involved_rules"]) for event in shield.last_solution.conflict_events if event["kind"] == "rule_rule"]
        self.assertIn({"C03:0", "C03:1"}, pairs)
        self.assertIn("C03:1", shield.last_solution.relaxed_constraints)

    def test_extension_templates_expose_note2_conflicts(self):
        cases = [
            (communication_intent_conflict_scenario(3), SimConfig(c2_timeout=0), {0: Action(Mode.CRITICAL), 1: Action(Mode.COVER), 2: Action(Mode.HOLD)}, {"F01:0", "C05:0"}),
            (capability_dwell_conflict_scenario(3), SimConfig(mode_dwell_steps=2), {0: Action(Mode.COVER), 1: Action(Mode.HOLD), 2: Action(Mode.HOLD)}, {"F03:0"}),
            (risk_continuity_conflict_scenario(3), SimConfig(c2_timeout=20), {0: Action(Mode.HOLD), 1: Action(Mode.COVER), 2: Action(Mode.HOLD)}, {"M05:0", "F06:1"}),
        ]
        for scenario, cfg, nominal, expected in cases:
            env = AircraftEnv(3, cfg)
            observations = env.reset(scenario)
            shield = ConflictAwareQPSafetyShield(cfg)
            shield.filter(observations, nominal)
            self.assertTrue(expected.intersection(set(shield.last_solution.belief_report.violations)))


if __name__ == "__main__":
    unittest.main()
