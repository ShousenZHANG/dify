"""API moderation retains legacy coercion and unconstrained extension input values."""

from pydantic import BaseModel, Field


class ModerationInputParams(BaseModel):
    app_id: str = ""
    inputs: dict[str, object] = Field(default_factory=dict)
    query: str = ""


class ModerationOutputParams(BaseModel):
    app_id: str = ""
    text: str
