class CubeDomainError(Exception):
    """Base exception for Rubik's cube domain errors."""


class InvalidFaceletCountError(CubeDomainError):
    """Raised when the total number of stickers on the cube is not 54."""


class InvalidColorDistributionError(CubeDomainError):
    """Raised when any color does not appear exactly 9 times."""


class DuplicateCenterColorError(CubeDomainError):
    """Raised when cube center stickers do not have 6 unique colors."""


class InvalidFaceOrderError(CubeDomainError):
    """Raised when required faces are missing or improperly defined."""
