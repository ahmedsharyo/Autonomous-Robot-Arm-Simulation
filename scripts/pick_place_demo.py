import json
from robot_arm.simulation import ArmSimulation
from robot_arm.manipulation import PickPlaceTask

def main():
    sim=ArmSimulation(); print(json.dumps(PickPlaceTask(sim).run(),indent=2))
if __name__=='__main__': main()
