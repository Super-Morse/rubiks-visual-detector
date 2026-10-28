import pytest

from app.domain.exceptions import (
    DuplicateCenterColorError,
    InvalidColorDistributionError,
    InvalidFaceOrderError,
)
from app.domain.models import Color, CubeState, FacePosition, FaceState
from app.infrastructure.validators import StandardCubeValidator


def create_solved_face(face: FacePosition, color: Color) -> FaceState:
    return FaceState(face=face, stickers=[color] * 9)


def create_valid_solved_cube() -> CubeState:
    return CubeState(
        faces={
            FacePosition.U: create_solved_face(FacePosition.U, Color.WHITE),
            FacePosition.R: create_solved_face(FacePosition.R, Color.RED),
            FacePosition.F: create_solved_face(FacePosition.F, Color.GREEN),
            FacePosition.D: create_solved_face(FacePosition.D, Color.YELLOW),
            FacePosition.L: create_solved_face(FacePosition.L, Color.ORANGE),
            FacePosition.B: create_solved_face(FacePosition.B, Color.BLUE),
        }
    )


class TestCubeValidator:
    def setup_method(self) -> None:
        self.validator = StandardCubeValidator()

    def test_valid_solved_cube_passes_validation(self) -> None:
        cube = create_valid_solved_cube()
        self.validator.validate(cube)

    def test_missing_face_raises_invalid_face_order(self) -> None:
        faces = {
            FacePosition.U: create_solved_face(FacePosition.U, Color.WHITE),
            FacePosition.R: create_solved_face(FacePosition.R, Color.RED),
            FacePosition.F: create_solved_face(FacePosition.F, Color.GREEN),
            FacePosition.D: create_solved_face(FacePosition.D, Color.YELLOW),
            FacePosition.L: create_solved_face(FacePosition.L, Color.ORANGE),
        }
        cube = CubeState(faces=faces)

        with pytest.raises(InvalidFaceOrderError, match="All 6 faces must be present"):
            self.validator.validate(cube)

    def test_invalid_color_counts_raises_error(self) -> None:
        cube = create_valid_solved_cube()
        # Make 10 white and 8 yellow stickers
        cube.faces[FacePosition.D].stickers[0] = Color.WHITE

        with pytest.raises(InvalidColorDistributionError) as exc_info:
            self.validator.validate(cube)
        assert "Each color must appear exactly 9 times" in str(exc_info.value)

    def test_duplicate_centers_raises_error(self) -> None:
        cube = create_valid_solved_cube()
        # Replace yellow center with white
        cube.faces[FacePosition.D].stickers[4] = Color.WHITE
        # Keep global sticker balance at 9 to isolate duplicate center verification
        cube.faces[FacePosition.U].stickers[0] = Color.YELLOW

        with pytest.raises(DuplicateCenterColorError) as exc_info:
            self.validator.validate(cube)
        assert "Face centers must have 6 unique colors" in str(exc_info.value)

    def test_to_kociemba_string_conversion(self) -> None:
        cube = create_valid_solved_cube()
        kociemba_str = cube.to_kociemba_string()

        expected = (
            "UUUUUUUUU"
            "RRRRRRRRR"
            "FFFFFFFFF"
            "DDDDDDDDD"
            "LLLLLLLLL"
            "BBBBBBBBB"
        )
        assert kociemba_str == expected
        assert len(kociemba_str) == 54
