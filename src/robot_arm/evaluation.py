"""Deterministic IK comparison and summary-statistic utilities."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from .ik import solve_dls,solve_optimization
from .kinematics import forward_kinematics

def summarize(values: list[float]) -> dict:
    a=np.asarray(values,float)
    return {'n':int(a.size),'mean':float(np.mean(a)),'median':float(np.median(a)),'std':float(np.std(a)),'p95':float(np.percentile(a,95))}

def benchmark_ik(trials: int=40, seed: int=17, output: str|Path|None=None) -> dict:
    """Compare two solvers on identical targets generated from feasible configs."""
    rng=np.random.default_rng(seed); records=[]
    for i in range(trials):
        q=rng.uniform([-.8,-.8,-1,-.8,-.8,-1],[.8,.8,1,.8,.8,1]); target=forward_kinematics(q)[0]
        q0=q+rng.normal(0,.2,6)
        for name,solver in [('dls',solve_dls),('least_squares',solve_optimization)]:
            r=solver(target,q0); records.append({'trial':i,'seed':seed,'method':name,'success':r.success,'error_m':r.position_error,'runtime_ms':r.runtime_s*1000,'iterations':r.iterations})
    summary={m:{k:summarize([r[k] for r in records if r['method']==m and (k not in ('error_m','runtime_ms','iterations') or r['success'])]) for k in ('error_m','runtime_ms','iterations')}|{'success_rate':float(np.mean([r['success'] for r in records if r['method']==m]))} for m in ('dls','least_squares')}
    if output:
        p=Path(output); p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
        p.with_suffix('.summary.json').write_text(json.dumps({'seed':seed,'trials':trials,'summary':summary},indent=2))
    return {'seed':seed,'trials':trials,'summary':summary,'records':records}
