import numpy as np
from src.robot_arm.ik import solve_dls,solve_optimization,LOWER,UPPER
from src.robot_arm.kinematics import forward_kinematics

def test_both_ik_solve_reachable_target():
    q=np.array([.3,-.4,.6,.2,-.5,.1]); target=forward_kinematics(q)[0]; q0=q+np.array([.08,-.05,.06,0,.03,-.04])
    for solver in (solve_dls,solve_optimization):
        result=solver(target,q0); assert result.success; assert result.position_error<1e-4

def test_unreachable_target_fails_gracefully():
    result=solve_dls(np.array([4.,0.,4.]),np.zeros(6),max_iterations=40)
    assert not result.success and np.isfinite(result.position_error)

def test_joint_bounds_defined():
    assert np.all(LOWER<UPPER)
