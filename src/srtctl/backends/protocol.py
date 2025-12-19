# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Protocol definition for backend implementations.

This module defines the interface that all backend configurations must implement.
Each backend is responsible for:
1. Allocating logical endpoints (serving units)
2. Converting endpoints to physical processes
3. Starting and managing those processes
"""

from typing import Dict, List, Protocol, Sequence

from srtctl.core.endpoints import Endpoint, Process
from srtctl.core.process_registry import NamedProcesses
from srtctl.core.runtime import RuntimeContext


class BackendProtocol(Protocol):
    """Protocol that all backend configurations must implement.

    This allows different serving frameworks (SGLang, vLLM, TRT-LLM) to be
    used interchangeably while providing backend-specific process launching.

    Example usage:
        backend = SGLangBackend(config)
        endpoints = backend.allocate_endpoints(common, nodes)
        processes = backend.endpoints_to_processes(endpoints)
        running = backend.start_processes(processes, runtime)
    """

    def allocate_endpoints(
        self,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
        available_nodes: Sequence[str],
    ) -> List[Endpoint]:
        """Allocate logical endpoints based on backend-specific logic.

        Args:
            num_prefill: Number of prefill workers
            num_decode: Number of decode workers
            num_agg: Number of aggregated workers
            gpus_per_prefill: GPUs per prefill worker
            gpus_per_decode: GPUs per decode worker
            gpus_per_agg: GPUs per agg worker
            gpus_per_node: GPUs per node
            available_nodes: Tuple of available node hostnames

        Returns:
            List of Endpoint objects with GPU allocations
        """
        ...

    def endpoints_to_processes(
        self,
        endpoints: List[Endpoint],
        base_port: int = 8081,
    ) -> List[Process]:
        """Convert logical endpoints to physical processes.

        Backend-specific mapping:
        - SGLang: 1 process per node (uses all GPUs on node)
        - vLLM: 1 process per GPU
        - TRT-LLM: 1 process per GPU

        Args:
            endpoints: List of logical endpoints
            base_port: Base port for DYN_SYSTEM_PORT assignment

        Returns:
            List of Process objects with individual process details
        """
        ...

    def start_processes(
        self,
        processes: List[Process],
        runtime: RuntimeContext,
        environment: Dict[str, str],
    ) -> NamedProcesses:
        """Start all processes for this backend.

        This is the main entry point for launching workers. Each backend
        implements this differently based on how the serving framework
        expects to be invoked.

        Args:
            processes: List of Process objects to start
            runtime: RuntimeContext with paths and node information
            environment: Additional environment variables

        Returns:
            Dict mapping process names to ManagedProcess objects
        """
        ...

