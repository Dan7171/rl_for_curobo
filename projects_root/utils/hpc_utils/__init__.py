"""
High Performance Computing utilities for CuRobo.

This package provides client-server implementations for distributing
heavy computational workloads across multiple machines.
"""

from .mpc_solver_api import MpcSolverApi, RemoteAttribute

__all__ = ['MpcSolverApi', 'RemoteAttribute'] 