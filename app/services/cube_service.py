from app.domain.interfaces import ICubeValidator
from app.domain.models import CubeState, FacePosition, FaceState


class CubeScannerService:
    """Assembles scanned faces into a CubeState and verifies structural integrity."""

    def __init__(self, validator: ICubeValidator) -> None:
        self._validator = validator

    def assemble_and_validate(
        self, face_states: dict[FacePosition, FaceState]
    ) -> CubeState:
        """Assembles 6 faces into a CubeState and verifies structural integrity.

        Raises:
            CubeDomainError: If the cube violates physical 3x3 constraints.
        """
        cube = CubeState(faces=face_states)
        self._validator.validate(cube)
        return cube
