# config/__init__.py
import os

# XLA flags for better GPU performance
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"

import jax

# Persistent JAX compilation cache
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)  # only cache slow compilations

print("Using performance configuration.")