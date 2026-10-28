class CubeDomainException(Exception):
    """Base exception for Rubik's cube domain errors."""


class InvalidFaceletCountError(CubeDomainException):
    """Raised when the total number of stickers on the cube is not 54."""


class InvalidColorDistributionError(CubeDomainException):
    """Raised when any color does not appear exactly 9 times."""


class DuplicateCenterColorError(CubeDomainException):
    """Raised when cube center stickers do not have 6 unique colors."""


class InvalidFaceOrderError(CubeDomainException):
    """Raised when required faces are missing or improperly defined."""
