"""Baseline MLP correctness: run the UNMODIFIED mlp kernel in nki.simulate,
compare vs torch SwiGLU reference. Small dims that satisfy H%128==0 and I_shard fits banks."""
import os
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE","trn2")
import numpy as np
import nki, nki.language as nl
from nkilib_src.nkilib.core.mlp.mlp import mlp
import inspect
# discover call signature quickly
params = list(inspect.signature(mlp).parameters)
print("mlp params[:8]:", params[:8])

# small single-core-friendly config: H=256 (2*128), I=512 (1 int_tile, 8//1=... fine), B=1,S=128
B,S,H,I = 1,128,256,512
np.random.seed(0)
hidden = np.random.randn(B,S,H).astype(np.float32)*0.1
Wg = np.random.randn(H,I).astype(np.float32)*0.1
Wu = np.random.randn(H,I).astype(np.float32)*0.1
Wd = np.random.randn(I,H).astype(np.float32)*0.1

def silu(x): return x/(1+np.exp(-x))
x2 = hidden.reshape(B*S,H)
ref = (silu(x2@Wg)*(x2@Wu))@Wd   # (B*S,H)

try:
    out = nki.simulate(mlp)(hidden.astype(np.float32), Wg.astype(np.float32), Wu.astype(np.float32), Wd.astype(np.float32))
    out = np.asarray(out).reshape(B*S,H)
    err = np.abs(out-ref).max()
    rel = err/ (np.abs(ref).max()+1e-9)
    print(f"OUT shape {out.shape}  max_abs_err={err:.4e}  rel={rel:.4e}")
    print("PASS" if rel < 2e-2 else "FAIL (rel too high)")
except Exception as e:
    import traceback; traceback.print_exc()
    print("SIM FAILED:", str(e)[:300])
