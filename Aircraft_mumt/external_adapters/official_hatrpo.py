"""Aircraft adapter for marlbenchmark/on-policy's official HATRPO core."""

from dataclasses import dataclass
from functools import wraps

from .official_happo import OfficialHAPPOAdapter, OfficialHAPPOConfig, _load_official_core


@dataclass
class OfficialHATRPOConfig(OfficialHAPPOConfig):
    kl_threshold: float = 0.01
    line_search_steps: int = 10
    accept_ratio: float = 0.5


class OfficialHATRPOAdapter(OfficialHAPPOAdapter):
    """Run HATRPO with the same Aircraft collection protocol as HAPPO."""

    component_name = "HATRPO"

    def _load_core(self):
        _load_official_core(self.source)
        from onpolicy.algorithms.utils.distributions import DiagGaussian

        # Upstream's HATRPO Box branch passes an always-None availability
        # argument to DiagGaussian.forward, whose signature only accepts x.
        # Accept and ignore that argument without changing the computation.
        if not getattr(DiagGaussian.forward, "_aircraft_hatrpo_compatible", False):
            original_forward = DiagGaussian.forward

            @wraps(original_forward)
            def compatible_forward(module, x, available_actions=None):
                return original_forward(module, x)

            compatible_forward._aircraft_hatrpo_compatible = True
            DiagGaussian.forward = compatible_forward

        from onpolicy.algorithms.hatrpo.policy import HATRPO_Policy
        from onpolicy.algorithms.hatrpo.hatrpo_trainer import HATRPO
        from onpolicy.utils.separated_buffer import SeparatedReplayBuffer

        return HATRPO_Policy, HATRPO, SeparatedReplayBuffer

    def _official_args(self):
        args = super()._official_args()
        args.algorithm_name = "hatrpo"
        args.kl_threshold = self.config.kl_threshold
        args.ls_step = self.config.line_search_steps
        args.accept_ratio = self.config.accept_ratio
        return args
