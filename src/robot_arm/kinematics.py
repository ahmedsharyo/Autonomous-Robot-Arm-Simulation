"""Independent serial-chain forward kinematics and geometric Jacobians."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class JointSpec:
    """A revolute joint axis and fixed translation from the prior joint."""
    offset: tuple[float, float, float]
    axis: tuple[float, float, float]

# The sequence exactly mirrors the six revolute joints in assets/robot.xml.
SPECS = (
    JointSpec((0, 0, .07), (0, 0, 1)),
    JointSpec((0, 0, .12), (0, 1, 0)),
    JointSpec((.25, 0, 0), (0, 1, 0)),
    JointSpec((.23, 0, 0), (1, 0, 0)),
    JointSpec((.09, 0, 0), (0, 1, 0)),
    JointSpec((.07, 0, 0), (1, 0, 0)),
)
BASE = np.array([0., 0., .39])
TOOL = np.array([.025, 0., 0.])

def _rot(axis: FloatArray, angle: float) -> FloatArray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])

def forward_kinematics(q: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return world-frame end-effector position and rotation for six angles."""
    q = np.asarray(q, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("q must contain six finite joint angles")
    p = BASE.copy(); R = np.eye(3)
    for angle, spec in zip(q, SPECS):
        p = p + R @ np.asarray(spec.offset, dtype=float)
        R = R @ _rot(np.asarray(spec.axis, dtype=float), float(angle))
    return p + R @ TOOL, R

def geometric_jacobian(q: FloatArray) -> FloatArray:
    """Return the 6x6 world-frame geometric Jacobian [linear; angular]."""
    q = np.asarray(q, dtype=float)
    if q.shape != (6,): raise ValueError("q must contain six joint angles")
    p = BASE.copy(); R = np.eye(3); origins=[]; axes=[]
    for angle, spec in zip(q, SPECS):
        p = p + R @ np.asarray(spec.offset, dtype=float)
        axis = R @ np.asarray(spec.axis, dtype=float)
        origins.append(p.copy()); axes.append(axis)
        R = R @ _rot(np.asarray(spec.axis, dtype=float), float(angle))
    pe = p + R @ TOOL
    J = np.zeros((6, 6))
    for i, (o, a) in enumerate(zip(origins, axes)):
        J[:3, i] = np.cross(a, pe-o); J[3:, i] = a
    return J

def finite_difference_jacobian(q: FloatArray, epsilon: float = 1e-6) -> FloatArray:
    """Numerically estimate the position Jacobian for validation."""
    q = np.asarray(q, dtype=float); J = np.empty((3, 6))
    for i in range(6):
        d = np.zeros(6); d[i] = epsilon
        J[:, i] = (forward_kinematics(q+d)[0]-forward_kinematics(q-d)[0])/(2*epsilon)
    return J
