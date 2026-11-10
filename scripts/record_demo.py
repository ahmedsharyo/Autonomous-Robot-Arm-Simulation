from pathlib import Path
from src.robot_arm.simulation import ArmSimulation
from src.robot_arm.visualization import record_scene

def main():
    p=Path('results/videos/src.robot_arm_demo.mp4');p.parent.mkdir(parents=True,exist_ok=True);record_scene(ArmSimulation(),p);print(p)
if __name__=='__main__': main()
