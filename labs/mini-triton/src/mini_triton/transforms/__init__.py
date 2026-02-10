"""
Mini-Triton Transformations (Optimization Passes).

This module contains optimization passes that transform the IR.
"""

from mini_triton.transforms.canonicalize import canonicalize
from mini_triton.transforms.loop_gen import generate_loops

__all__ = ["canonicalize", "generate_loops"]
