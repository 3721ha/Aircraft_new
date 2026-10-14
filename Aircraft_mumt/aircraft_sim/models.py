from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class Mode(str, Enum):
    HOLD = "hold"
    SUPPORT = "support"
    COVER = "cover"
    RECOVER = "recover"
    EXIT = "exit"
    CRITICAL = "critical"


class TargetKind(str, Enum):
    """Semantic type of the object an aircraft is assigned to act on."""

    NONE = "none"
    TEAMMATE = "teammate"
    INTRUDER = "intruder"
    PROTECTED_ASSET = "protected_asset"
    RESOURCE = "resource"


@dataclass
class Action:
    mode: Mode = Mode.HOLD
    turn: float = 0.0
    climb: float = 0.0
    acceleration: float = 0.0
    target: Optional[int] = None
    target_kind: TargetKind | str = TargetKind.NONE

    def clipped(self) -> "Action":
        return Action(
            self.mode,
            max(-1.0, min(1.0, self.turn)),
            max(-1.0, min(1.0, self.climb)),
            max(-1.0, min(1.0, self.acceleration)),
            self.target,
            self.target_kind,
        )


@dataclass
class AircraftState:
    uid: int
    position: Tuple[float, float, float]
    speed: float = 220.0
    heading: float = 0.0
    climb_angle: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    angle_of_attack: float = 0.0
    load_factor: float = 1.0
    energy: float = 0.8
    resource: float = 1.0
    capability: float = 1.0
    role: Mode = Mode.HOLD
    c2_connected: bool = True
    sensor_degraded: bool = False
    id_confidence: float = 1.0
    alive: bool = True
    last_message_step: int = 0
    pending_support_since: Optional[int] = None
    intervention_count: int = 0
    mode_age: int = 0
    dwell_violation: bool = False
    authorization_valid: bool = True
    intent_shared: bool = True
    data_age: int = 0
    task_progress: float = 0.0
    handover_confirmed: bool = True
    intervention_window: list = field(default_factory=list)
    source_disagreement: float = 0.0
    warning_pending: bool = False
    warning_age: int = 0
    support_requested: bool = False
    assessment_due: bool = False
    last_task_action_step: Optional[int] = None
    replan_requested: bool = False
    task_authority: bool = True
    hazard_active: bool = False
    safe_dwell: int = 0
    hazard_previous: bool = False

    def copy(self) -> "AircraftState":
        return AircraftState(**self.__dict__)


@dataclass
class StepResult:
    observations: dict
    reward: float
    done: bool
    info: dict = field(default_factory=dict)
