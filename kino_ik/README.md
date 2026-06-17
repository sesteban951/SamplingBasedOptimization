# kino_ik

Kinodynamic IK pipeline: given an SRB trajectory (CoM, momentum, contacts), finds a full 29-DOF joint trajectory that satisfies the whole-body centroidal dynamics.

## Files

| File | Role |
|------|------|
| `kino_nlp.py` | Core NLP — builds and solves the CasADi Opti problem |
| `pipeline_kino_ik.py` | End-to-end driver: load SRB results → run kino IK → save/plot |
| `validate_kino.py` | Post-solve validation (residual checks, joint limit checks) |

## Running

```bash
conda run -n env_sbo python kino_ik/pipeline_kino_ik.py
# Force a solver rebuild:
conda run -n env_sbo python kino_ik/pipeline_kino_ik.py --rebuild
```

## Solver cache

The NLP build step (symbolics + codegen) is expensive (~30–60 s).  
`KinoNLP` serialises the compiled `ca.Function` to `.kino_cache/kino_solve_<hash>.casadi` and reloads it on subsequent runs.

### What's in the cache key

| Factor | Invalidates cache? |
|--------|--------------------|
| `_CACHE_VERSION` string | Yes |
| `dt_vec`, SRB trajectory arrays | Yes |
| **Cost weights** (`w_ke`, `w_config`, …) | **No** — weights are runtime parameters; tuning them reuses the cache |
| URDF / robot model | Indirectly (same URDF assumed; model isn't hashed) |

### When you must force a rebuild

> **The trap:** changing a weight value is safe and never requires a rebuild.  
> Changing the *form* of a cost term does require one.

| Change | Rebuild needed? | How to handle |
|--------|-----------------|---------------|
| Tune a weight (e.g. `w_config = 1e2`) | No | Just run normally |
| Add / remove a weight key | Yes | Bump `_CACHE_VERSION` or `--rebuild` |
| Change a cost expression (e.g. alter KE term, change which nodes `w_config` applies to) | **Yes** | Bump `_CACHE_VERSION` (e.g. `v2 → v3`) or pass `--rebuild` once |
| Add / remove a constraint | Yes | Bump `_CACHE_VERSION` or `--rebuild` |
| Change `dt_vec` or SRB reference data | Yes | Automatic (hashed into cache key) |

`--rebuild` is a one-shot override; it does not change the version string, so the old cache file is left on disk.  
Bumping `_CACHE_VERSION` in `kino_nlp.py` is the persistent fix and is preferred when the structural change should be permanent.
