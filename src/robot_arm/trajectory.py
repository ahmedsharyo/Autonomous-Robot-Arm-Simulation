"""Rest-to-rest trajectory generation in Cartesian or joint space."""
from __future__ import annotations
import numpy as np

def quintic_scaling(t: float, duration: float) -> tuple[float,float,float]:
    """Return position, velocity and acceleration scale for quintic rest-to-rest motion."""
    if duration<=0: raise ValueError('duration must be positive')
    u=float(np.clip(t/duration,0,1)); s=10*u**3-15*u**4+6*u**5
    sd=(30*u**2-60*u**3+30*u**4)/duration
    sdd=(60*u-180*u**2+120*u**3)/duration**2
    return s,sd,sdd

def interpolate(start: np.ndarray, goal: np.ndarray, t: float, duration: float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Evaluate smooth quintic interpolation and its first two derivatives."""
    a=np.asarray(start,float); b=np.asarray(goal,float)
    if a.shape!=b.shape: raise ValueError('start and goal shapes differ')
    s,sd,sdd=quintic_scaling(t,duration); d=b-a
    return a+s*d,sd*d,sdd*d
