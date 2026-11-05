"""Position inverse kinematics: damped least squares and bounded least squares."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import least_squares
from numpy.typing import NDArray
from .kinematics import forward_kinematics, geometric_jacobian
FloatArray=NDArray[np.float64]
LOWER=np.array([-3.14,-2.4,-2.6,-3.14,-2.4,-3.14]); UPPER=-LOWER
@dataclass
class IKResult:
    q: FloatArray
    success: bool
    position_error: float
    iterations: int
    runtime_s: float
    message: str

def solve_dls(target: FloatArray, q0: FloatArray, *, damping: float=.04, gain: float=.8, tol: float=1e-4, max_iterations: int=250) -> IKResult:
    """Iterative damped least-squares position IK with bounded updates."""
    target=np.asarray(target,float); q=np.clip(np.asarray(q0,float),LOWER,UPPER).copy(); start=time.perf_counter()
    if target.shape!=(3,) or q.shape!=(6,): raise ValueError('target must be 3D and q0 must have six values')
    for k in range(max_iterations):
        e=target-forward_kinematics(q)[0]; err=float(np.linalg.norm(e))
        if err<tol: return IKResult(q,True,err,k,time.perf_counter()-start,'converged')
        J=geometric_jacobian(q)[:3]
        dq=J.T @ np.linalg.solve(J@J.T+damping*damping*np.eye(3),e)
        n=np.linalg.norm(dq)
        if n>.18: dq *= .18/n
        q=np.clip(q+gain*dq,LOWER+1e-5,UPPER-1e-5)
    err=float(np.linalg.norm(target-forward_kinematics(q)[0]))
    return IKResult(q,err<tol,err,max_iterations,time.perf_counter()-start,'iteration limit; target may be unreachable')

def solve_optimization(target: FloatArray, q0: FloatArray, *, tol: float=1e-4, max_nfev: int=500) -> IKResult:
    """Bounded SciPy least-squares position IK with preferred-posture regularization."""
    target=np.asarray(target,float); q0=np.clip(np.asarray(q0,float),LOWER+1e-8,UPPER-1e-8); start=time.perf_counter()
    if target.shape!=(3,) or q0.shape!=(6,): raise ValueError('target must be 3D and q0 must have six values')
    def residual(q): return np.r_[forward_kinematics(q)[0]-target, .002*(q-q0)]
    result=least_squares(residual,q0,bounds=(LOWER,UPPER),max_nfev=max_nfev,xtol=1e-10,ftol=1e-10,gtol=1e-10)
    err=float(np.linalg.norm(forward_kinematics(result.x)[0]-target))
    return IKResult(result.x,bool(result.success and err<tol),err,int(result.nfev),time.perf_counter()-start,str(result.message))
