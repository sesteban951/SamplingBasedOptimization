# config/__init__.py
import os
import jax

# XLA flags for better GPU performance
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"

# Persistent JAX compilation cache
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.jax_cache"))

print("Using performance configuration.")