"""OCPP 1.6-J integration: a central system that runs an evcharge policy live.

Needs the ``ocpp`` extra (``pip install "ev-smart-charging[ocpp]"``, which adds
``websockets``). :mod:`evcharge.ocpp.framing` and :mod:`evcharge.ocpp.messages`
have no network dependency. See ``docs/ocpp-demo.md``.
"""

from __future__ import annotations

try:
    import websockets  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        'evcharge.ocpp needs the "ocpp" extra: pip install "ev-smart-charging[ocpp]"'
    ) from exc

from evcharge.ocpp.central import (
    CentralSystem,
    CentralSystemConfig,
    Declared,
    SessionDefaults,
    Transaction,
    config_from_dict,
    load_config,
    serve,
)
from evcharge.ocpp.chargepoint import EV, ChargePointLog, ScaledClock, SimulatedChargePoint
from evcharge.ocpp.demo import DemoResult, run_demo
from evcharge.ocpp.framing import SUBPROTOCOL, Call, CallError, CallResult, ErrorCode, parse

__all__ = [
    "EV",
    "SUBPROTOCOL",
    "Call",
    "CallError",
    "CallResult",
    "CentralSystem",
    "CentralSystemConfig",
    "ChargePointLog",
    "Declared",
    "DemoResult",
    "ErrorCode",
    "ScaledClock",
    "SessionDefaults",
    "SimulatedChargePoint",
    "Transaction",
    "config_from_dict",
    "load_config",
    "parse",
    "run_demo",
    "serve",
]
