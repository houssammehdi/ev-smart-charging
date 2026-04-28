"""OCPP-J 1.6 message framing (no network code).

OCPP-J sends every message as a JSON array over a WebSocket negotiated with
the ``ocpp1.6`` subprotocol:

* CALL ``[2, "<messageId>", "<Action>", {payload}]``
* CALLRESULT ``[3, "<messageId>", {payload}]``
* CALLERROR ``[4, "<messageId>", "<errorCode>", "<errorDescription>", {errorDetails}]``

A CALLRESULT or CALLERROR answers the CALL with the same message id, and each
side may have only one CALL outstanding on a connection at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

SUBPROTOCOL = "ocpp1.6"
"""WebSocket subprotocol of OCPP 1.6 over JSON."""

MAX_MESSAGE_ID_LENGTH = 36
"""Longest message id the specification allows (a UUID string)."""


class MessageType(IntEnum):
    """First element of every OCPP-J message."""

    CALL = 2
    CALLRESULT = 3
    CALLERROR = 4


class ErrorCode(StrEnum):
    """CALLERROR codes of OCPP-J 1.6 (spelling as in the specification)."""

    NOT_IMPLEMENTED = "NotImplemented"
    NOT_SUPPORTED = "NotSupported"
    INTERNAL_ERROR = "InternalError"
    PROTOCOL_ERROR = "ProtocolError"
    SECURITY_ERROR = "SecurityError"
    FORMATION_VIOLATION = "FormationViolation"
    PROPERTY_CONSTRAINT_VIOLATION = "PropertyConstraintViolation"
    OCCURRENCE_CONSTRAINT_VIOLATION = "OccurenceConstraintViolation"
    TYPE_CONSTRAINT_VIOLATION = "TypeConstraintViolation"
    GENERIC_ERROR = "GenericError"


class FramingError(ValueError):
    """A message that is not valid OCPP-J.

    Attributes:
        code: The CALLERROR code to answer with.
        message_id: Id of the offending message if it could be read (else ``None``).
    """

    def __init__(self, code: ErrorCode, description: str, message_id: str | None = None) -> None:
        super().__init__(description)
        self.code = code
        self.message_id = message_id


@dataclass(frozen=True)
class Call:
    """A request."""

    message_id: str
    action: str
    payload: dict[str, object] = field(default_factory=dict)

    def encode(self) -> str:
        """JSON text of the message."""
        return json.dumps([int(MessageType.CALL), self.message_id, self.action, self.payload])


@dataclass(frozen=True)
class CallResult:
    """A successful response."""

    message_id: str
    payload: dict[str, object] = field(default_factory=dict)

    def encode(self) -> str:
        """JSON text of the message."""
        return json.dumps([int(MessageType.CALLRESULT), self.message_id, self.payload])


@dataclass(frozen=True)
class CallError:
    """An error response."""

    message_id: str
    code: ErrorCode
    description: str = ""
    details: dict[str, object] = field(default_factory=dict)

    def encode(self) -> str:
        """JSON text of the message."""
        kind = int(MessageType.CALLERROR)
        return json.dumps([kind, self.message_id, str(self.code), self.description, self.details])


Message = Call | CallResult | CallError


def _object(value: object, what: str, message_id: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FramingError(ErrorCode.FORMATION_VIOLATION, f"{what} must be an object", message_id)
    return {str(k): v for k, v in value.items()}


def parse(text: str | bytes) -> Message:
    """Decode one OCPP-J message.

    Raises:
        FramingError: with the error code to answer with, and the message id
            when it could be read.
    """
    try:
        data = json.loads(text)
    except (ValueError, UnicodeDecodeError):
        raise FramingError(ErrorCode.PROTOCOL_ERROR, "message is not valid JSON") from None
    if not isinstance(data, list) or len(data) < 3:
        raise FramingError(ErrorCode.PROTOCOL_ERROR, "message must be a JSON array of 3 to 5 items")
    kind, message_id = data[0], data[1]
    if not isinstance(message_id, str) or not message_id:
        raise FramingError(ErrorCode.PROTOCOL_ERROR, "message id must be a non-empty string")
    if len(message_id) > MAX_MESSAGE_ID_LENGTH:
        raise FramingError(
            ErrorCode.PROTOCOL_ERROR, "message id longer than 36 characters", message_id
        )
    if isinstance(kind, bool) or kind not in (2, 3, 4):
        raise FramingError(ErrorCode.PROTOCOL_ERROR, f"unknown message type {kind!r}", message_id)
    if kind == MessageType.CALL:
        if len(data) != 4 or not isinstance(data[2], str) or not data[2]:
            raise FramingError(
                ErrorCode.PROTOCOL_ERROR, "CALL must be [2, id, action, payload]", message_id
            )
        return Call(message_id, data[2], _object(data[3], "CALL payload", message_id))
    if kind == MessageType.CALLRESULT:
        if len(data) != 3:
            raise FramingError(
                ErrorCode.PROTOCOL_ERROR, "CALLRESULT must be [3, id, payload]", message_id
            )
        return CallResult(message_id, _object(data[2], "CALLRESULT payload", message_id))
    if len(data) != 5 or not isinstance(data[2], str) or not isinstance(data[3], str):
        raise FramingError(
            ErrorCode.PROTOCOL_ERROR,
            "CALLERROR must be [4, id, code, description, details]",
            message_id,
        )
    try:
        code = ErrorCode(data[2])
    except ValueError:
        code = ErrorCode.GENERIC_ERROR
    return CallError(message_id, code, data[3], _object(data[4], "CALLERROR details", message_id))
