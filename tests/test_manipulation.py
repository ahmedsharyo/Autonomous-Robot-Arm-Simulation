from robot_arm.manipulation import TaskState

def test_state_machine_declares_expected_order():
    assert [TaskState.APPROACH,TaskState.DESCEND,TaskState.GRASP,TaskState.LIFT,TaskState.TRANSFER,TaskState.PLACE,TaskState.RELEASE,TaskState.RETREAT]
