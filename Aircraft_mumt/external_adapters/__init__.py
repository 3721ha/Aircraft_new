"""Adapters for immutable third-party research implementations."""

from .official_happo import OfficialHAPPOAdapter, OfficialHAPPOConfig
from .official_hatrpo import OfficialHATRPOAdapter, OfficialHATRPOConfig
from .official_gcbfplus import (
    GCBFPlusDependencyError,
    OfficialGCBFPlusConfig,
    OfficialGCBFPlusShield,
    dependencies_available as gcbfplus_dependencies_available,
)
from .official_mappo import OfficialMAPPOAdapter, OfficialMAPPOConfig
from .official_macpo import OfficialMACPOAdapter, OfficialMACPOConfig
from .official_mappo_lagrangian import OfficialMAPPOLagrangianAdapter, OfficialMAPPOLagrangianConfig
from .official_mat import OfficialMATAdapter, OfficialMATConfig

__all__ = [
    "OfficialHAPPOAdapter",
    "OfficialHAPPOConfig",
    "OfficialHATRPOAdapter",
    "OfficialHATRPOConfig",
    "GCBFPlusDependencyError",
    "OfficialGCBFPlusConfig",
    "OfficialGCBFPlusShield",
    "gcbfplus_dependencies_available",
    "OfficialMACPOAdapter",
    "OfficialMACPOConfig",
    "OfficialMAPPOAdapter",
    "OfficialMAPPOConfig",
    "OfficialMAPPOLagrangianAdapter",
    "OfficialMAPPOLagrangianConfig",
    "OfficialMATAdapter",
    "OfficialMATConfig",
]
