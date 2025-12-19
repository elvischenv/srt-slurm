"""
srtctl - Benchmark submission framework for distributed serving workloads.

This package provides Python-first orchestration for LLM inference benchmarks
on SLURM clusters, supporting SGLang with prefill/decode disaggregation.

Key modules:
- core.config: Configuration loading and validation
- core.runtime: RuntimeContext for computed paths and values
- core.endpoints: Endpoint and Process dataclasses for worker topology
- core.process_registry: Process lifecycle management
- backends.sglang: SGLang backend implementation
- cli.submit: Job submission interface
- cli.do_sweep: Main orchestration script

Usage:
    # Submit with legacy mode (existing Jinja templates)
    srtctl apply -f config.yaml

    # Submit with new orchestrator (Python-controlled)
    srtctl apply -f config.yaml --use-orchestrator
"""

__version__ = "0.2.0"

# Core modules
from .core.config import load_config, get_srtslurm_setting
from .core.runtime import Nodes, RuntimeContext, get_slurm_job_id, get_hostname_ip
from .core.endpoints import Endpoint, Process, allocate_endpoints, endpoints_to_processes
from .core.process_registry import (
    ManagedProcess,
    NamedProcesses,
    ProcessRegistry,
)

# Backend
from .core.backend import SGLangBackend

__all__ = [
    # Version
    "__version__",
    # Config
    "load_config",
    "get_srtslurm_setting",
    # Runtime
    "Nodes",
    "RuntimeContext",
    "get_slurm_job_id",
    "get_hostname_ip",
    # Endpoints
    "Endpoint",
    "Process",
    "allocate_endpoints",
    "endpoints_to_processes",
    # Process management
    "ManagedProcess",
    "NamedProcesses",
    "ProcessRegistry",
    # Backend
    "SGLangBackend",
]
