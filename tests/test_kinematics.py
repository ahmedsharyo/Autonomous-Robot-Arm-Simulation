import numpy as np
import pytest
from src.robot_arm.kinematics import forward_kinematics,geometric_jacobian,finite_difference_jacobian

def test_jacobian_matches_finite_difference():
    q=np.array([.3,-.4,.6,.2,-.5,.1])
    assert np.allclose(geometric_jacobian(q)[:3],finite_difference_jacobian(q),atol=1e-7)

def test_fk_validates_configuration():
    with pytest.raises(ValueError): forward_kinematics(np.zeros(5))
