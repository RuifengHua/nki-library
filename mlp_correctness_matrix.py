"""Correctness matrix harness — validates mlp vs torch SwiGLU across the configs
that matter for the (C) restructure. Reusable for baseline AND post-optimization."""
import os
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE","trn2")
import numpy as np, nki
from nkilib_src.nkilib.core.mlp.mlp import mlp

def silu(x): return x/(1+np.exp(-x))

def run(name, B,S,H,I, lnc, seed=0, rtol=2e-2):
    np.random.seed(seed)
    hidden=(np.random.randn(B,S,H)*0.1).astype(np.float32)
    Wg=(np.random.randn(H,I)*0.1).astype(np.float32)
    Wu=(np.random.randn(H,I)*0.1).astype(np.float32)
    Wd=(np.random.randn(I,H)*0.1).astype(np.float32)
    x=hidden.reshape(B*S,H); ref=(silu(x@Wg)*(x@Wu))@Wd
    try:
        kfn = mlp[lnc] if lnc>1 else mlp
        out=np.asarray(nki.simulate(kfn)(hidden,Wg,Wu,Wd)).reshape(B*S,H)
        rel=np.abs(out-ref).max()/(np.abs(ref).max()+1e-9)
        ok = rel<rtol
        print(f"[{'PASS' if ok else 'FAIL'}] {name:22} B{B} S{S} H{H} I{I} lnc{lnc}  rel={rel:.3e}")
        return ok
    except Exception as e:
        print(f"[ERR ] {name:22} {str(e)[:120]}")
        return False

# configs: small mirrors of the profiled sweep. I_shard = I/lnc.
# narrow-I (2-subtile, healthy): I_shard/512 <= 4
# wide-I  (1-subtile, starved) : I_shard/512 >= 6   <-- the (C) target
results=[]
results.append(run("narrow_I512_lnc1",   1,128, 256, 512, 1))   # int_tiles=1 -> 8 subtiles cap
results.append(run("mid_I2048_lnc1",     1,256, 512, 2048,1))    # int_tiles=4 -> 2 subtiles
results.append(run("wide_I4096_lnc1",    1,256, 512, 4096,1))    # int_tiles=8 -> 1 subtile (STARVED target)
results.append(run("wide_I3072_lnc1",    1,256, 512, 3072,1))    # int_tiles=6 -> 1 subtile, NOT div by 4
results.append(run("shard_lnc2_I4096",   1,256,1024, 4096,2))    # I_shard=2048 int_tiles=4
print(f"\n{sum(results)}/{len(results)} PASS")
