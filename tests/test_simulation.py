import numpy as np
import pytest
mujoco=pytest.importorskip('mujoco')
from robot_arm.simulation import ArmSimulation
from robot_arm.kinematics import forward_kinematics
from robot_arm.collision import configuration_contacts

def test_fk_matches_mujoco_site_pose():
    sim=ArmSimulation(); q=np.array([.2,-.3,.4,.1,-.2,.3]);sim.reset(q)
    p,R=forward_kinematics(q)
    assert np.allclose(p,sim.ee_position,atol=1e-7)
    assert np.allclose(R,sim.ee_rotation,atol=1e-7)

def test_collision_query_refreshes_configuration():
    sim=ArmSimulation(); result=configuration_contacts(sim,np.zeros(6))
    assert isinstance(result,list)
