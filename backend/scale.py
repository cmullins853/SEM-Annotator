import math


def compute_scale(pixel_distance: float, known_length_um: float) -> float:
    """Returns µm/px from a drawn line measurement."""
    if pixel_distance <= 0:
        raise ValueError("Pixel distance must be positive")
    if known_length_um <= 0:
        raise ValueError("Known length must be positive")
    return known_length_um / pixel_distance


def pixel_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two image-space points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def pixel_distance_horizontal(x1: float, x2: float) -> float:
    """Horizontal-only distance for Shift-constrained lines."""
    return abs(x2 - x1)
