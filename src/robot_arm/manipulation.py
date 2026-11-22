"""Waypoint-driven pick-and-place state machine with physical grasp verification."""
from __future__ import annotations
from enum import Enum,auto
import numpy as np
from .simulation import ArmSimulation
from .ik import solve_dls
from .control import torque_tracking
from .trajectory import interpolate

class TaskState(Enum): APPROACH=auto(); DESCEND=auto(); GRASP=auto(); LIFT=auto(); TRANSFER=auto(); PLACE=auto(); RELEASE=auto(); RETREAT=auto(); DONE=auto(); FAILED=auto()

class PickPlaceTask:
    """Execute a conservative Cartesian waypoint sequence using IK and PD torque."""
    def __init__(self, sim: ArmSimulation, duration: float=1.4, control_hz: int=100):
        self.sim=sim; self.duration=duration; self.control_hz=control_hz; self.state=TaskState.APPROACH; self.history=[]
    def _move(self, p: np.ndarray) -> bool:
        ik=solve_dls(p,self.sim.q)
        if not ik.success: return False
        q0=self.sim.q.copy(); steps=max(2,int(self.duration*self.control_hz)); dt=1/self.control_hz
        for i in range(steps+1):
            qd,qdv,_=interpolate(q0,ik.q,i*dt,self.duration)
            tau=torque_tracking(self.sim.q,self.sim.qvel,qd,qdv)
            self.sim.set_torque(tau); self.sim.step(max(1,round(dt/self.sim.model.opt.timestep)))
        return np.linalg.norm(self.sim.ee_position-p)<.025
    def run(self, object_position: np.ndarray|None=None, destination: np.ndarray=np.array([.58,-.24,.425])) -> dict:
        obj=self.sim.object_position.copy() if object_position is None else np.asarray(object_position,float).copy()
        # Gripper closes along local y; this model's tool axis is horizontal, so physical
        # grasping is only considered successful when actual finger/object contact occurs.
        waypoints=[(TaskState.APPROACH,obj+np.array([0,0,.16])),(TaskState.DESCEND,obj+np.array([0,0,.045]))]
        for state,p in waypoints:
            self.state=state; self.history.append(state.name)
            if not self._move(p): self.state=TaskState.FAILED; return self._report(obj,destination,False)
        self.state=TaskState.GRASP; self.history.append(self.state.name)
        # Close gripper controls are position actuator indices after six torque motors.
        self.sim.data.ctrl[6]=0.0; self.sim.data.ctrl[7]=0.0
        self.sim.step(300)
        contacts=self.sim.contacts()
        grasp=any('object_geom' in pair and ('finger_left' in pair or 'finger_right' in pair) for pair in contacts)
        if not grasp: self.state=TaskState.FAILED; return self._report(obj,destination,False)
        for state,p in [(TaskState.LIFT,obj+np.array([0,0,.18])),(TaskState.TRANSFER,destination+np.array([0,0,.18])),(TaskState.PLACE,destination+np.array([0,0,.04]))]:
            self.state=state; self.history.append(state.name)
            if not self._move(p): self.state=TaskState.FAILED; return self._report(obj,destination,False)
        self.state=TaskState.RELEASE; self.history.append(self.state.name)
        # position actuators open the fingers back to their initialized values
        self.sim.data.ctrl[6]=.04; self.sim.data.ctrl[7]=-.04; self.sim.step(300)
        self.state=TaskState.RETREAT; self.history.append(self.state.name)
        if not self._move(destination+np.array([0,0,.18])): self.state=TaskState.FAILED; return self._report(obj,destination,False)
        final=self.sim.object_position
        placed=np.linalg.norm(final[:2]-destination[:2])<.07 and abs(final[2]-.425)<.05
        self.state=TaskState.DONE if placed else TaskState.FAILED; self.history.append(self.state.name)
        return self._report(obj,destination,placed)
    def _report(self,obj,dest,success):
        return {'success':bool(success),'state':self.state.name,'states':self.history.copy(),'initial_object':obj.tolist(),
                'destination':np.asarray(dest).tolist(),'final_object':self.sim.object_position.tolist(),'contacts':self.sim.contacts()}
