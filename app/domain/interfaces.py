from abc import ABC, abstractmethod

from app.domain.models import CubeState


class ICubeValidator(ABC):
    """Contract for verifying physical and structural validity of a cube state."""

    @abstractmethod
    def validate(self, cube_state: CubeState) -> None:
        """Validates cube state.

        Raises:
            CubeDomainException: When cube violates physical or geometric rules.
        """
        pass
