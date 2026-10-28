from collections import Counter

from app.domain.exceptions import (
    DuplicateCenterColorError,
    InvalidColorDistributionError,
    InvalidFaceOrderError,
)
from app.domain.interfaces import ICubeValidator
from app.domain.models import CubeState, FacePosition

REQUIRED_FACES = {
    FacePosition.U,
    FacePosition.R,
    FacePosition.F,
    FacePosition.D,
    FacePosition.L,
    FacePosition.B,
}


class StandardCubeValidator(ICubeValidator):
    """Validates structural and physical constraints of a 3x3 Rubik's cube."""

    def validate(self, cube_state: CubeState) -> None:
        self._validate_faces_presence(cube_state)
        self._validate_centers(cube_state)
        self._validate_color_distribution(cube_state)

    def _validate_faces_presence(self, cube_state: CubeState) -> None:
        if set(cube_state.faces.keys()) != REQUIRED_FACES:
            expected = [f.value for f in REQUIRED_FACES]
            raise InvalidFaceOrderError(f"All 6 faces must be present: {expected}")

    def _validate_centers(self, cube_state: CubeState) -> None:
        centers = [face.center_color for face in cube_state.faces.values()]
        if len(set(centers)) != 6:
            detected = [c.value for c in centers]
            raise DuplicateCenterColorError(
                f"Face centers must have 6 unique colors. Received: {detected}"
            )

    def _validate_color_distribution(self, cube_state: CubeState) -> None:
        all_stickers = [
            color
            for face in cube_state.faces.values()
            for color in face.stickers
        ]

        counts = Counter(all_stickers)
        anomalies = {
            color.value: count
            for color, count in counts.items()
            if count != 9
        }

        if anomalies:
            raise InvalidColorDistributionError(
                f"Each color must appear exactly 9 times. Anomalies detected: {anomalies}"
            )
