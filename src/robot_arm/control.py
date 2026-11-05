"""Kinematic command generation and torque-level tracking control."""
from __future__ import annotations
import numpy as np
from .kinematics import forward_kinematics, geometric_jacobian
from .ik import LOWER,UPPER

def cartesian_velocity(q: np.ndarray, target: np.ndarray, gain: float=2.0, max_joint_speed: float=1.2, damping: float=.05) -> np.ndarray:
    """Generate joint velocity command via damped position Jacobian inverse."""
    e=np.asarray(target)-forward_kinematics(q)[0]
    J=geometric_jacobian(q)[:3]
    qdot=J.T@np.linalg.solve(J@J.T+damping**2*np.eye(3),gain*e)
    return np.clip(qdot,-max_joint_speed,max_joint_speed)

def torque_tracking(q: np.ndarray, qvel: np.ndarray, q_des: np.ndarray, qvel_des: np.ndarray|None=None, kp: float=80., kd: float=16.) -> np.ndarray:
    """Computed joint-space PD torque; output must be clipped to actuator bounds."""
    qvel_des=np.zeros(6) if qvel_des is None else np.asarray(qvel_des)
    return kp*(np.asarray(q_des)-q)+kd*(qvel_des-np.asarray(qvel))

def valid_configuration(q: np.ndarray, margin: float=0.) -> bool:
    """Check finite values and declared joint bounds."""
    q=np.asarray(q)
    return q.shape==(6,) and bool(np.all(np.isfinite(q)) and np.all(q>=LOWER+margin) and np.all(q<=UPPER-margin))
