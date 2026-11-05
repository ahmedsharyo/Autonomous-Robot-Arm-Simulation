"""Configuration-space collision queries using MuJoCo's collision engine."""
from __future__ import annotations
import numpy as np
import mujoco
from .simulation import ArmSimulation

def configuration_contacts(sim: ArmSimulation, q: np.ndarray) -> list[tuple[str,str]]:
    """Apply q and refresh MuJoCo collision state before returning contacts."""
    sim.data.qpos[sim._qadr]=np.asarray(q,float)
    mujoco.mj_forward(sim.model,sim.data)
    return sim.contacts()

def has_unwanted_contact(contacts: list[tuple[str,str]], allowed: tuple[str,...]=('finger_left','finger_right','object_geom')) -> bool:
    """Conservatively flag any contact not involving explicitly allowed grasp geoms."""
    return any(not any(name in pair for name in allowed) for pair in contacts)
