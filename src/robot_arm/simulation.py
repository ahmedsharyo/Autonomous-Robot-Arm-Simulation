"""MuJoCo simulation wrapper with deterministic reset and semantic accessors."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / 'assets' / 'robot.xml'
JOINT_NAMES = tuple(f'joint{i}' for i in range(1, 7))

class ArmSimulation:
    """Small MuJoCo wrapper; controller and simulation stepping remain separate."""
    def __init__(self, model_path: str | Path = MODEL_PATH, timestep: float | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        if timestep is not None: self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self._qadr = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in JOINT_NAMES])
        self._dadr = np.array([self.model.jnt_dofadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in JOINT_NAMES])
        self._act = np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,f'motor{i}') for i in range(1,7)])
        self._ee = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'ee')
        self._object = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'object')
        self.reset()
    def reset(self, q: np.ndarray | None = None) -> None:
        """Reset physics and object pose to a deterministic state."""
        mujoco.mj_resetData(self.model, self.data)
        if q is not None: self.data.qpos[self._qadr] = np.asarray(q,dtype=float)
        self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'object_free')]:][:3] = [.43,0,.425]
        mujoco.mj_forward(self.model,self.data)
    @property
    def q(self) -> np.ndarray: return self.data.qpos[self._qadr].copy()
    @property
    def qvel(self) -> np.ndarray: return self.data.qvel[self._dadr].copy()
    @property
    def ee_position(self) -> np.ndarray: return self.data.site_xpos[self._ee].copy()
    @property
    def ee_rotation(self) -> np.ndarray: return self.data.site_xmat[self._ee].reshape(3,3).copy()
    @property
    def object_position(self) -> np.ndarray: return self.data.xpos[self._object].copy()
    def set_torque(self, tau: np.ndarray) -> None: self.data.ctrl[self._act] = np.clip(tau,-35,35)
    def step(self, n: int = 1) -> None:
        for _ in range(n): mujoco.mj_step(self.model,self.data)
    def contacts(self) -> list[tuple[str,str]]:
        """Return geom-name pairs for active contacts."""
        out=[]
        for i in range(self.data.ncon):
            c=self.data.contact[i]
            out.append((mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) or str(c.geom1),mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2) or str(c.geom2)))
        return out
