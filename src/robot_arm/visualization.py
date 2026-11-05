"""Plotting and headless MP4 recording helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import mujoco
from .simulation import ArmSimulation

def plot_ik_benchmark(result: dict, path: str|Path) -> None:
    methods=('dls','least_squares'); fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,key,title in zip(axes,('error_m','runtime_ms'),('Endpoint error (m)','Solver runtime (ms)')):
        ax.boxplot([[r[key] for r in result['records'] if r['method']==m] for m in methods],tick_labels=methods); ax.set_title(title);ax.grid(alpha=.25)
    fig.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=160);plt.close(fig)

def record_scene(sim: ArmSimulation, path: str|Path, frames: int=240, width: int=640, height: int=480) -> None:
    """Record a deterministic passive scene with MuJoCo's offscreen renderer."""
    renderer=mujoco.Renderer(sim.model,height=height,width=width); writer=imageio.get_writer(path,fps=30,codec='libx264')
    try:
        for _ in range(frames):
            renderer.update_scene(sim.data,camera='track');writer.append_data(renderer.render());sim.step(4)
    finally: writer.close();renderer.close()
