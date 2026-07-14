from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class FileItem(BaseModel):
    filename: str
    content_type: str
    data: bytes
    path: Optional[str] = None


class SchemaPayload(BaseModel):
    schema_name: str
    fields: list[dict]

    @classmethod
    def parse_json(cls, raw_schema: str) -> "SchemaPayload":
        cleaned = (
            raw_schema.replace("\u00a0", " ")
            .replace("\u200b", "")
            .replace("\ufeff", "")
        )
        return cls.model_validate_json(cleaned)