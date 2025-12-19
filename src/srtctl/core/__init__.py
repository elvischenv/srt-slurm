# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core modules for srtctl.

This package contains:
- config: Configuration loading and validation
- runtime: RuntimeContext for computed paths and values
- endpoints: Endpoint and Process dataclasses
- process_registry: Process lifecycle management
- utils: Helper functions (srun, wait_for_port, etc.)
"""

from .config import load_config, get_srtslurm_setting
from .endpoints import Endpoint, Process, allocate_endpoints, endpoints_to_processes
from .process_registry import (
    ManagedProcess,
    NamedProcesses,
    ProcessRegistry,
    setup_signal_handlers,
    start_process_monitor,
)
from .runtime import Nodes, RuntimeContext, get_slurm_job_id, get_hostname_ip

__all__ = [
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
    "setup_signal_handlers",
    "start_process_monitor",
]

