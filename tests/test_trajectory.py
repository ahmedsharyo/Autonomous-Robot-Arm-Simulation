import numpy as np
from src.robot_arm.trajectory import interpolate

def test_quintic_trajectory_endpoint_continuity():
    a=np.array([0.,1.]);b=np.array([2.,-1.])
    p0,v0,acc0=interpolate(a,b,0,2);p1,v1,acc1=interpolate(a,b,2,2)
    assert np.allclose(p0,a) and np.allclose(p1,b)
    assert np.allclose(v0,0) and np.allclose(v1,0)
    assert np.allclose(acc0,0) and np.allclose(acc1,0)
