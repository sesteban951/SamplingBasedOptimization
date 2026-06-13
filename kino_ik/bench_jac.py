"""
Microbenchmark for the kino-dynamics speed bottleneck.

Measures per-eval cost of the centroidal-momentum constraint Jacobian
(the `nlp_jac_g` hotspot, ~86 ms/call in the full solve) under:
  (a) plain cpin AD          — what the current NLP uses
  (b) JIT-compiled cpin AD   — same expression, native code
and reports the speedup, plus a check that hg-value and Jac match.

Run:
  conda run -n env_sbo python kino_ik/bench_jac.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF  = os.path.join(_REPO, "models", "g1", "g1_29dof_rev_1_0.urdf")

model = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
nq, nv = model.nq, model.nv
cmodel = cpin.Model(model)
cdata  = cmodel.createData()

q_sx = ca.SX.sym("q", nq)
v_sx = ca.SX.sym("v", nv)
cpin.ccrba(cmodel, cdata, q_sx, v_sx)
hg = ca.vertcat(cdata.hg.linear, cdata.hg.angular)

x_sx = ca.vertcat(q_sx, v_sx)              # stacked input (71,)
Jhg  = ca.jacobian(hg, x_sx)               # 6 x 71  (AD-once Jacobian)

# ---- build the two function variants -------------------------------------
def build(jit):
    opts = {}
    if jit:
        opts = {"jit": True, "compiler": "shell",
                "jit_options": {"flags": ["-O2"], "verbose": False},
                "jit_temp_suffix": False}
    f_hg = ca.Function("f_hg", [q_sx, v_sx], [hg], opts)
    f_J  = ca.Function("f_J",  [q_sx, v_sx], [Jhg], opts)
    return f_hg, f_J

print("[bench] building plain (no-jit) functions...")
t0 = time.time(); hg_plain, J_plain = build(False); print(f"  build {time.time()-t0:.2f}s")

jit_ok = True
try:
    print("[bench] building JIT functions (compiling C)...")
    t0 = time.time(); hg_jit, J_jit = build(True); print(f"  build {time.time()-t0:.2f}s")
except Exception as e:
    jit_ok = False
    print(f"  JIT build FAILED: {e}")

# ---- sample configurations (warm, valid) ---------------------------------
rng = np.random.default_rng(0)
M = 2000
qs = np.zeros((M, nq)); vs = np.zeros((M, nv))
for i in range(M):
    q = pin.randomConfiguration(model)
    q[:3] = rng.normal(size=3) * 0.3
    qs[i] = q
    vs[i] = rng.normal(size=nv) * 0.5

# correctness: JIT vs plain
if jit_ok:
    e_hg = max(np.abs(np.array(hg_plain(qs[i], vs[i])) - np.array(hg_jit(qs[i], vs[i]))).max()
               for i in range(20))
    e_J  = max(np.abs(np.array(J_plain(qs[i], vs[i])) - np.array(J_jit(qs[i], vs[i]))).max()
               for i in range(20))
    print(f"[bench] JIT vs plain agreement: hg {e_hg:.2e}, Jac {e_J:.2e}")

def timeit(f, label):
    f(qs[0], vs[0])                        # warm
    t0 = time.time()
    for i in range(M):
        f(qs[i], vs[i])
    dt = (time.time() - t0) / M * 1e3      # ms/eval
    print(f"  {label:22s} {dt:8.3f} ms/eval")
    return dt

print(f"\n[bench] per-eval timing over {M} samples:")
j_plain = timeit(J_plain, "Jacobian (plain)")
hgp     = timeit(hg_plain, "hg value (plain)")
if jit_ok:
    j_jit = timeit(J_jit, "Jacobian (JIT)")
    hgj   = timeit(hg_jit, "hg value (JIT)")
    print(f"\n[bench] JIT speedup:  Jacobian {j_plain/j_jit:.2f}x   hg {hgp/hgj:.2f}x")

# project to full-solve scale: jac_g evaluates ~ (2N momentum + ...) per call.
N = 75
print(f"\n[bench] scale to {N} momentum-balance nodes (2 ccrba-Jac per interval):")
print(f"  plain ~{2*N*j_plain:.0f} ms/full-Jac-eval")
if jit_ok:
    print(f"  JIT   ~{2*N*j_jit:.0f} ms/full-Jac-eval   (baseline solve had ~86 ms*... see log)")
