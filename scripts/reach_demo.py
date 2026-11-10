import numpy as np
from src.robot_arm.simulation import ArmSimulation
from src.robot_arm.ik import solve_dls
from src.robot_arm.control import torque_tracking
from src.robot_arm.trajectory import interpolate

def main():
    sim=ArmSimulation(); target=np.array([.48,.18,.68]); result=solve_dls(target,sim.q)
    print(f'IK success={result.success}, endpoint error={result.position_error:.4f} m, solver={result.runtime_s*1e3:.2f} ms')
    if not result.success: return
    q0=sim.q.copy(); dt=.01; duration=3.
    for i in range(int(duration/dt)+1):
        qd,qdv,_=interpolate(q0,result.q,i*dt,duration); sim.set_torque(torque_tracking(sim.q,sim.qvel,qd,qdv)); sim.step(5)
    print(f'actual endpoint={sim.ee_position}, error={np.linalg.norm(sim.ee_position-target):.4f} m')
if __name__=='__main__': main()
