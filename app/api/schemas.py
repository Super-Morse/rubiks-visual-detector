from pydantic import BaseModel, Field

from app.domain.models import Color, FacePosition


class FacePayload(BaseModel):
    """Payload representing a single scanned face submitted by client."""

    face: FacePosition
    stickers: list[Color] = Field(..., min_length=9, max_length=9)


class ValidateCubeRequest(BaseModel):
    """Request payload containing all 6 scanned faces."""

    faces: list[FacePayload] = Field(..., min_length=6, max_length=6)


class CubeValidationSuccess(BaseModel):
    """Successful response with validated cube state and Kociemba solver string."""

    is_valid: bool = True
    kociemba_string: str = Field(..., min_length=54, max_length=54)
    detected_faces: dict[FacePosition, list[Color]]
    message: str = "Cube is structurally valid and ready for solver."
