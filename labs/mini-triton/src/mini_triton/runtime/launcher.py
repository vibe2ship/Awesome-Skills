"""
Kernel Launcher for Mini-Triton.

This module provides utilities for launching kernels on different backends.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class LaunchConfig:
    """
    Configuration for launching a kernel.

    Attributes:
        grid: Grid dimensions (number of blocks in each dimension)
        block_size: Block size (for backends that need it)
        shared_mem: Shared memory size in bytes
    """
    grid: Tuple[int, ...]
    block_size: Tuple[int, ...] = (1, 1, 1)
    shared_mem: int = 0

    def __post_init__(self):
        # Normalize to 3D
        if len(self.grid) == 1:
            self.grid = (self.grid[0], 1, 1)
        elif len(self.grid) == 2:
            self.grid = (self.grid[0], self.grid[1], 1)


def launch_kernel(
    kernel: Callable,
    config: LaunchConfig,
    *args,
    **kwargs
) -> None:
    """
    Launch a compiled kernel.

    This function executes the kernel for each block in the grid.
    For the NumPy backend, this means executing the kernel function
    in a loop over program IDs.

    Args:
        kernel: The compiled kernel function
        config: Launch configuration
        *args: Kernel arguments (arrays, scalars)
        **kwargs: Additional keyword arguments
    """
    # TODO:
    # For NumPy backend:
    # 1. Iterate over grid dimensions
    # 2. Call kernel with program IDs and arguments
    #
    # for pid_x in range(config.grid[0]):
    #     for pid_y in range(config.grid[1]):
    #         for pid_z in range(config.grid[2]):
    #             kernel(pid_x, pid_y, pid_z, *args, **kwargs)
    raise NotImplementedError("TODO: Implement launch_kernel")


def compute_grid(n: int, block_size: int) -> Tuple[int]:
    """
    Compute 1D grid size for n elements with given block size.

    Args:
        n: Number of elements
        block_size: Elements per block

    Returns:
        Grid dimensions as 1-tuple

    Example:
        compute_grid(1000, 128)  # Returns (8,) since ceil(1000/128) = 8
    """
    # TODO: Return ((n + block_size - 1) // block_size,)
    raise NotImplementedError("TODO: Implement compute_grid")


def compute_grid_2d(m: int, n: int, block_m: int, block_n: int) -> Tuple[int, int]:
    """
    Compute 2D grid size for m x n elements.

    Args:
        m: First dimension size
        n: Second dimension size
        block_m: Block size in first dimension
        block_n: Block size in second dimension

    Returns:
        Grid dimensions as 2-tuple
    """
    # TODO: Return (ceil(m/block_m), ceil(n/block_n))
    raise NotImplementedError("TODO: Implement compute_grid_2d")
