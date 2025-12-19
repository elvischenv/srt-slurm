# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SGLang backend implementation.

This module provides SGLang-specific process launching logic,
replacing the scattered logic in scripts/worker_setup/*.
"""

import logging
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

from srtctl.core.endpoints import (
    Endpoint,
    Process,
    WorkerMode,
    allocate_endpoints,
    endpoints_to_processes,
)
from srtctl.core.process_registry import ManagedProcess, NamedProcesses
from srtctl.core.runtime import RuntimeContext, get_hostname_ip

logger = logging.getLogger(__name__)


@dataclass
class SGLangBackend:
    """SGLang backend for launching prefill/decode/agg workers.

    This class encapsulates all SGLang-specific configuration and process
    launching logic, implementing the BackendProtocol.

    Attributes:
        model_path: Path to the model
        served_model_name: Name to serve the model as
        prefill_config: SGLang config dict for prefill workers
        decode_config: SGLang config dict for decode workers
        agg_config: SGLang config dict for aggregated workers
        shared_config: Config shared across all worker types
        use_sglang_router: Use sglang.launch_server instead of dynamo.sglang
    """

    model_path: Path
    served_model_name: str
    prefill_config: Dict[str, Any] = field(default_factory=dict)
    decode_config: Dict[str, Any] = field(default_factory=dict)
    agg_config: Dict[str, Any] = field(default_factory=dict)
    shared_config: Dict[str, Any] = field(default_factory=dict)
    use_sglang_router: bool = False

    @classmethod
    def from_job_config(cls, config: Dict[str, Any]) -> "SGLangBackend":
        """Create SGLangBackend from a JobConfig dict.

        Args:
            config: Validated JobConfig as dict

        Returns:
            Configured SGLangBackend instance
        """
        model_path = Path(config["model"]["path"])
        served_model_name = model_path.name

        # Extract SGLang config sections
        backend_config = config.get("backend", {})
        sglang_config = backend_config.get("sglang_config", {})

        prefill_config = {}
        decode_config = {}
        agg_config = {}

        if sglang_config:
            if sglang_config.get("prefill"):
                prefill_config = dict(sglang_config["prefill"])
            if sglang_config.get("decode"):
                decode_config = dict(sglang_config["decode"])
            if sglang_config.get("aggregated"):
                agg_config = dict(sglang_config["aggregated"])

        # Check frontend config for sglang router mode
        frontend_config = config.get("frontend", {})
        use_sglang_router = frontend_config.get("use_sglang_router", False)

        return cls(
            model_path=model_path,
            served_model_name=served_model_name,
            prefill_config=prefill_config,
            decode_config=decode_config,
            agg_config=agg_config,
            use_sglang_router=use_sglang_router,
        )

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
        """Allocate endpoints to nodes.

        Delegates to the core allocation function.
        """
        return allocate_endpoints(
            num_prefill=num_prefill,
            num_decode=num_decode,
            num_agg=num_agg,
            gpus_per_prefill=gpus_per_prefill,
            gpus_per_decode=gpus_per_decode,
            gpus_per_agg=gpus_per_agg,
            gpus_per_node=gpus_per_node,
            available_nodes=available_nodes,
        )

    def endpoints_to_processes(
        self,
        endpoints: List[Endpoint],
        base_port: int = 8081,
    ) -> List[Process]:
        """Convert endpoints to processes.

        For SGLang, we create one process per node in each endpoint.
        """
        return endpoints_to_processes(endpoints, base_port)

    def get_config_for_mode(self, mode: WorkerMode) -> Dict[str, Any]:
        """Get the merged config dict for a worker mode.

        Args:
            mode: "prefill", "decode", or "agg"

        Returns:
            Merged config dict (shared_config + mode-specific config)
        """
        base_config = dict(self.shared_config)

        if mode == "prefill":
            base_config.update(self.prefill_config)
        elif mode == "decode":
            base_config.update(self.decode_config)
        elif mode == "agg":
            base_config.update(self.agg_config)

        return base_config

    def build_worker_command(
        self,
        process: Process,
        endpoint_processes: List[Process],
        runtime: RuntimeContext,
        dump_config_path: Optional[Path] = None,
    ) -> List[str]:
        """Build the command to start an SGLang worker process.

        Args:
            process: The Process to start
            endpoint_processes: All processes in this endpoint (for multi-node setup)
            runtime: RuntimeContext with paths
            dump_config_path: Optional path to dump config JSON

        Returns:
            Command as list of strings
        """
        mode = process.endpoint_mode
        config = self.get_config_for_mode(mode)

        # Determine if multi-node
        endpoint_nodes = list(dict.fromkeys(p.node for p in endpoint_processes))
        is_multi_node = len(endpoint_nodes) > 1

        # Get leader IP for distributed init
        leader_node = endpoint_nodes[0]
        leader_ip = get_hostname_ip(leader_node)
        dist_init_port = 29500

        # Choose Python module
        if self.use_sglang_router:
            python_module = "sglang.launch_server"
        else:
            python_module = "dynamo.sglang"

        cmd = [
            "python3",
            "-m",
            python_module,
            "--model-path",
            str(runtime.model_path),
            "--served-model-name",
            self.served_model_name,
            "--host",
            "0.0.0.0",
        ]

        # Add disaggregation mode flag (not for agg mode)
        if mode != "agg" and not self.use_sglang_router:
            cmd.extend(["--disaggregation-mode", mode])

        # Add multi-node coordination flags
        if is_multi_node:
            node_rank = endpoint_nodes.index(process.node)
            cmd.extend([
                "--dist-init-addr",
                f"{leader_ip}:{dist_init_port}",
                "--nnodes",
                str(len(endpoint_nodes)),
                "--node-rank",
                str(node_rank),
            ])

        # Add config dump path
        if dump_config_path and not self.use_sglang_router:
            cmd.extend(["--dump-config-to", str(dump_config_path)])

        # Add all config flags
        cmd.extend(self._config_to_cli_args(config))

        return cmd

    def _config_to_cli_args(self, config: Dict[str, Any]) -> List[str]:
        """Convert config dict to CLI arguments.

        Args:
            config: Configuration dictionary

        Returns:
            List of CLI arguments
        """
        args: List[str] = []

        for key, value in sorted(config.items()):
            # Convert snake_case to kebab-case
            flag_name = key.replace("_", "-")

            if isinstance(value, bool):
                if value:
                    args.append(f"--{flag_name}")
            elif isinstance(value, list):
                args.append(f"--{flag_name}")
                args.extend(str(v) for v in value)
            elif value is not None:
                args.extend([f"--{flag_name}", str(value)])

        return args

    def start_processes(
        self,
        processes: List[Process],
        runtime: RuntimeContext,
        environment: Dict[str, str],
    ) -> NamedProcesses:
        """Start all SGLang processes.

        This is a placeholder - the actual process starting is done by
        the orchestrator using start_srun_process(). This method is here
        to satisfy the BackendProtocol interface and could be used for
        local testing.

        Args:
            processes: List of Process objects to start
            runtime: RuntimeContext with paths
            environment: Environment variables

        Returns:
            Dict mapping process names to ManagedProcess objects
        """
        # Group processes by endpoint
        from collections import defaultdict

        grouped: Dict[tuple, List[Process]] = defaultdict(list)
        for process in processes:
            key = (process.endpoint_mode, process.endpoint_index)
            grouped[key].append(process)

        result: NamedProcesses = {}

        for endpoint_key, endpoint_processes in grouped.items():
            mode, index = endpoint_key

            for process in endpoint_processes:
                # Build command
                dump_path = (
                    runtime.log_dir
                    / f"{mode}_config_{index}_{process.node}.json"
                )
                cmd = self.build_worker_command(
                    process=process,
                    endpoint_processes=endpoint_processes,
                    runtime=runtime,
                    dump_config_path=dump_path,
                )

                process_name = f"{mode}_{index}_{process.node}"
                logger.info(
                    "Would start %s: %s",
                    process_name,
                    shlex.join(cmd),
                )

                # In actual use, the orchestrator calls start_srun_process()
                # This is just for interface compliance
                result[process_name] = None  # type: ignore

        return result


def build_sglang_command_from_config(
    mode: WorkerMode,
    config: Dict[str, Any],
    model_path: Path,
    served_model_name: str,
    leader_ip: str,
    dist_init_port: int,
    num_nodes: int,
    node_rank: int,
    use_sglang_router: bool = False,
    dump_config_path: Optional[Path] = None,
) -> List[str]:
    """Build an SGLang command from configuration.

    This is a standalone function for use in templates or scripts.

    Args:
        mode: Worker mode (prefill, decode, agg)
        config: SGLang config dict for this mode
        model_path: Path to model
        served_model_name: Model name to serve
        leader_ip: IP of the leader node
        dist_init_port: Port for distributed init
        num_nodes: Total nodes in this endpoint
        node_rank: This node's rank
        use_sglang_router: Use sglang.launch_server instead of dynamo.sglang
        dump_config_path: Optional path to dump config

    Returns:
        Command as list of strings
    """
    python_module = "sglang.launch_server" if use_sglang_router else "dynamo.sglang"

    cmd = [
        "python3",
        "-m",
        python_module,
        "--model-path",
        str(model_path),
        "--served-model-name",
        served_model_name,
        "--host",
        "0.0.0.0",
    ]

    # Add disaggregation mode
    if mode != "agg" and not use_sglang_router:
        cmd.extend(["--disaggregation-mode", mode])

    # Add multi-node flags
    if num_nodes > 1:
        cmd.extend([
            "--dist-init-addr",
            f"{leader_ip}:{dist_init_port}",
            "--nnodes",
            str(num_nodes),
            "--node-rank",
            str(node_rank),
        ])

    # Add dump config
    if dump_config_path and not use_sglang_router:
        cmd.extend(["--dump-config-to", str(dump_config_path)])

    # Add config flags
    for key, value in sorted(config.items()):
        flag_name = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{flag_name}")
        elif isinstance(value, list):
            cmd.append(f"--{flag_name}")
            cmd.extend(str(v) for v in value)
        elif value is not None:
            cmd.extend([f"--{flag_name}", str(value)])

    return cmd

