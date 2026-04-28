"""Payload checks for the OCPP 1.6 messages the central system handles.

Only what the controller relies on is validated: required fields and their
JSON types, and the enumerations it interprets. A failed check becomes the
CALLERROR code the specification prescribes (``FormationViolation`` for a
missing field, ``TypeConstraintViolation`` for a wrong type,
``PropertyConstraintViolation`` for a value outside its range).
"""

from __future__ import annotations

from datetime import UTC, datetime

from evcharge.ocpp.framing import ErrorCode

CHARGE_POINT_STATUS = frozenset(
    {
        "Available",
        "Preparing",
        "Charging",
        "SuspendedEVSE",
        "SuspendedEV",
        "Finishing",
        "Reserved",
        "Unavailable",
        "Faulted",
    }
)
"""ChargePointStatus values of OCPP 1.6."""


class PayloadError(ValueError):
    """A payload that does not match the message's schema."""

    def __init__(self, code: ErrorCode, description: str) -> None:
        super().__init__(description)
        self.code = code


def require(payload: dict[str, object], key: str) -> object:
    """Return a required field or raise ``FormationViolation``."""
    if key not in payload:
        raise PayloadError(ErrorCode.FORMATION_VIOLATION, f"missing required field {key!r}")
    return payload[key]


def as_int(value: object, key: str) -> int:
    """An integer field (JSON booleans are not integers)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise PayloadError(ErrorCode.TYPE_CONSTRAINT_VIOLATION, f"{key} must be an integer")
    return value


def as_str(value: object, key: str, max_length: int | None = None) -> str:
    """A string field, optionally with the specification's maximum length."""
    if not isinstance(value, str):
        raise PayloadError(ErrorCode.TYPE_CONSTRAINT_VIOLATION, f"{key} must be a string")
    if max_length is not None and len(value) > max_length:
        raise PayloadError(
            ErrorCode.PROPERTY_CONSTRAINT_VIOLATION, f"{key} longer than {max_length} characters"
        )
    return value


def as_time(value: object, key: str) -> datetime:
    """An ISO 8601 date-time field; ``Z`` and a missing offset both mean UTC."""
    text = as_str(value, key)
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise PayloadError(
            ErrorCode.TYPE_CONSTRAINT_VIOLATION, f"{key} is not an ISO 8601 date-time"
        ) from None
    return when if when.tzinfo is not None else when.replace(tzinfo=UTC)


def as_list(value: object, key: str) -> list[object]:
    """An array field."""
    if not isinstance(value, list):
        raise PayloadError(ErrorCode.TYPE_CONSTRAINT_VIOLATION, f"{key} must be an array")
    return value


def as_object(value: object, key: str) -> dict[str, object]:
    """An object field."""
    if not isinstance(value, dict):
        raise PayloadError(ErrorCode.TYPE_CONSTRAINT_VIOLATION, f"{key} must be an object")
    return {str(k): v for k, v in value.items()}


def energy_register_wh(meter_value: list[object]) -> float | None:
    """Latest ``Energy.Active.Import.Register`` reading in Wh, if any.

    ``measurand`` defaults to that register and ``unit`` to Wh, as in the
    specification; readings in kWh are converted. Other measurands are ignored.
    """
    reading: float | None = None
    for i, entry in enumerate(meter_value):
        mv = as_object(entry, f"meterValue[{i}]")
        for j, raw in enumerate(as_list(require(mv, "sampledValue"), "sampledValue")):
            sv = as_object(raw, f"sampledValue[{j}]")
            measurand = sv.get("measurand", "Energy.Active.Import.Register")
            if measurand != "Energy.Active.Import.Register":
                continue
            text = as_str(require(sv, "value"), "value")
            try:
                value = float(text)
            except ValueError:
                raise PayloadError(
                    ErrorCode.TYPE_CONSTRAINT_VIOLATION, f"sampled value {text!r} is not a number"
                ) from None
            unit = sv.get("unit", "Wh")
            if unit == "kWh":
                value *= 1000.0
            elif unit != "Wh":
                raise PayloadError(
                    ErrorCode.PROPERTY_CONSTRAINT_VIOLATION,
                    f"unit {unit!r} is not an energy unit",
                )
            reading = value
    return reading
