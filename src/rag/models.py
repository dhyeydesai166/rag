from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

STRING_FIELDS = (
    "id",
    "text",
    "policy",
    "version",
    "section",
    "heading_path",
    "parent_id",
    "source",
    "embed_text",
    "text_sha256",
    "embed_sha256",
)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    policy: str
    version: str
    section: str
    heading_path: str
    parent_id: str
    source: str
    embed_text: str
    text_sha256: str
    embed_sha256: str
    word_count: int
    ordinal: int | None = None

    @field_validator(*STRING_FIELDS)
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("blank")
        return value


class Claim(BaseModel):
    text: str
    chunk_id: str


class Answer(BaseModel):
    status: Literal["answered", "not_in_sources", "conflicting"]
    claims: list[Claim]
