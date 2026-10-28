from enum import Enum
from pydantic import BaseModel, Field


class Color(str, Enum):
    WHITE = "white"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    RED = "red"
    ORANGE = "orange"


class FacePosition(str, Enum):
    U = "U"  # Up (Standard: White)
    R = "R"  # Right (Standard: Red)
    F = "F"  # Front (Standard: Green)
    D = "D"  # Down (Standard: Yellow)
    L = "L"  # Left (Standard: Orange)
    B = "B"  # Back (Standard: Blue)


# Standard WCA default color-to-face mapping (BOY convention)
DEFAULT_COLOR_TO_FACE: dict[Color, FacePosition] = {
    Color.WHITE: FacePosition.U,
    Color.RED: FacePosition.R,
    Color.GREEN: FacePosition.F,
    Color.YELLOW: FacePosition.D,
    Color.ORANGE: FacePosition.L,
    Color.BLUE: FacePosition.B,
}


class FaceState(BaseModel):
    """Represents a single 3x3 face (9 stickers).

    Sticker indexing convention (row-major order):
    [0, 1, 2]
    [3, 4, 5]  -> Index 4 represents the fixed center sticker
    [6, 7, 8]
    """

    face: FacePosition
    stickers: list[Color] = Field(..., min_length=9, max_length=9)

    @property
    def center_color(self) -> Color:
        return self.stickers[4]


class CubeState(BaseModel):
    """Represents full cube state with 6 faces in WCA order: U, R, F, D, L, B."""

    faces: dict[FacePosition, FaceState]

    def to_kociemba_string(
        self,
        color_mapping: dict[Color, FacePosition] | None = None,
    ) -> str:
        """Converts 54 stickers to a standard 54-char Kociemba notation string.

        If color_mapping is not provided, it is dynamically resolved
        from the center sticker of each face.
        """
        if color_mapping is None:
            color_mapping = {
                face_state.center_color: face_pos
                for face_pos, face_state in self.faces.items()
            }

        order = [
            FacePosition.U,
            FacePosition.R,
            FacePosition.F,
            FacePosition.D,
            FacePosition.L,
            FacePosition.B,
        ]

        result: list[str] = []
        for face_pos in order:
            face_state = self.faces[face_pos]
            for color in face_state.stickers:
                result.append(color_mapping[color].value)

        return "".join(result)
