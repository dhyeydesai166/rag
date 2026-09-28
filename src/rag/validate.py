from pydantic import ValidationError

from rag.logutil import log
from rag.models import Chunk


def _field_name(exc: ValidationError) -> str:
    loc = exc.errors()[0]["loc"]
    return str(loc[0]) if loc else "record"


def validate(record: dict) -> str | None:
    error = None
    for attempt in (1, 2):
        # TODO: Attempt 2 is futile, must be removed, check with compliance team if they need it for auditory reasons before removing
        try:
            Chunk.model_validate(record)
        except ValidationError as exc:
            field = _field_name(exc)
            error = f"missing field: {field}"
            log("validate", f"id={record.get('id')} attempt={attempt} {error}")
            continue
        log("validate", f"id={record.get('id')} attempt={attempt} pass")
        return None
    return error
