from abc import ABC, abstractmethod

from app.domain.models import CubeState, FaceScanResult


class ICubeValidator(ABC):
    """Contract for verifying physical and structural validity of a cube state."""

    @abstractmethod
    def validate(self, cube_state: CubeState) -> None:
        """Validates cube state.

        Raises:
            CubeDomainException: When cube violates physical or geometric rules.
        """
        pass


class IFaceDetector(ABC):
    """Contract for detecting 3x3 Rubik's cube face stickers on an image."""

    @abstractmethod
    def detect(self, image_bytes: bytes) -> FaceScanResult:
        """Processes raw image bytes and returns sorted 9-sticker face state.

        Raises:
            ValueError: If image cannot be decoded.
            RuntimeError: If exactly 9 face stickers cannot be resolved.
        """
        pass
