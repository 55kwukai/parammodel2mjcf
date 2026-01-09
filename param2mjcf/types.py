import numpy as np
from dataclasses import dataclass, field

@dataclass
class Pose:
    """Represents a spatial pose with position and quaternion rotation."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Quaternion in (w, x, y, z) format for MuJoCo
    quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
