from app.domain.models import Color, CubeState, FacePosition, FaceState
from app.infrastructure.validators import StandardCubeValidator
from app.services.cube_service import CubeScannerService


def make_solved_cube_dict() -> dict[FacePosition, FaceState]:
    return {
        FacePosition.U: FaceState(face=FacePosition.U, stickers=[Color.WHITE] * 9),
        FacePosition.R: FaceState(face=FacePosition.R, stickers=[Color.RED] * 9),
        FacePosition.F: FaceState(face=FacePosition.F, stickers=[Color.GREEN] * 9),
        FacePosition.D: FaceState(face=FacePosition.D, stickers=[Color.YELLOW] * 9),
        FacePosition.L: FaceState(face=FacePosition.L, stickers=[Color.ORANGE] * 9),
        FacePosition.B: FaceState(face=FacePosition.B, stickers=[Color.BLUE] * 9),
    }


def test_service_assemble_and_validate_success():
    """Tests that 6 valid faces are properly assembled and validated into a CubeState."""
    service = CubeScannerService(validator=StandardCubeValidator())
    faces = make_solved_cube_dict()
    cube = service.assemble_and_validate(faces)

    assert isinstance(cube, CubeState)
    assert cube.to_kociemba_string() == (
        "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    )
