import unittest

from aircraft_sim import Action, AircraftEnv, ConflictAwareQPSafetyShield, Mode, SimConfig
from aircraft_sim.scenarios import capability_dwell_conflict_scenario, protocol_conflict_scenario


class ComponentAblationTests(unittest.TestCase):
    def test_no_graph_keeps_rule_report_but_removes_graph_events(self):
        config = SimConfig(support_response_deadline=0)
        env = AircraftEnv(3, config)
        observations = env.reset(protocol_conflict_scenario(3))
        nominal = {i: Action(Mode.CRITICAL) for i in observations}

        full = ConflictAwareQPSafetyShield(config)
        full.filter(observations, nominal)
        flat = ConflictAwareQPSafetyShield(config, use_dependency_graph=False)
        flat.filter(observations, nominal)

        self.assertEqual(set(full.last_solution.belief_report.violations), set(flat.last_solution.belief_report.violations))
        self.assertTrue(full.last_solution.conflict_events)
        self.assertEqual(flat.last_solution.conflict_events, [])

    def test_gate_only_skips_continuous_solver(self):
        config = SimConfig(noise_position=0.0, noise_speed=0.0)
        env = AircraftEnv(3, config)
        observations = env.reset(capability_dwell_conflict_scenario(3))
        shield = ConflictAwareQPSafetyShield(config, use_continuous_qp=False)
        shield.filter(observations, {0: Action(Mode.COVER), 1: Action(Mode.HOLD), 2: Action(Mode.HOLD)})
        self.assertEqual(shield.last_solution.status, "gate_only")
        self.assertEqual(shield.last_solution.solver_agents, [])
        self.assertEqual(shield.last_solution.solver_components, [])


if __name__ == "__main__":
    unittest.main()
