# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime context and node configuration.

This module provides the single source of truth for all runtime values,
replacing scattered bash variables and Jinja templating with typed Python.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence
import os
import socket

if TYPE_CHECKING:
    from .schema import JobConfig


@dataclass(frozen=True)
class Nodes:
    """Node allocation for head, benchmark, and worker nodes.

    Attributes:
        head: Head node hostname (runs NATS, etcd, nginx)
        bench: Benchmark node hostname (runs the benchmark client)
        worker: Tuple of all worker node hostnames (prefill + decode)
    """

    head: str
    bench: str
    worker: tuple[str, ...]

    @classmethod
    def from_slurm(cls, benchmark_on_separate_node: bool = False) -> "Nodes":
        """Create Nodes from SLURM environment.

        Args:
            benchmark_on_separate_node: If True, first node is benchmark-only,
                                        second is head, rest are workers.
        """
        nodelist = get_slurm_nodelist()
        if not nodelist:
            raise RuntimeError("SLURM_NODELIST not set - are we running in SLURM?")

        if benchmark_on_separate_node:
            if len(nodelist) < 2:
                raise ValueError(
                    "benchmark_on_separate_node requires at least 2 nodes"
                )
            bench = nodelist[0]
            head = nodelist[1]
            worker = tuple(nodelist[1:])
        else:
            head = nodelist[0]
            bench = head
            worker = tuple(nodelist[:])

        return cls(head=head, bench=bench, worker=worker)


@dataclass(frozen=True)
class RuntimeContext:
    """Runtime context with all computed values.

    This is the single source of truth for all runtime values and paths.
    All paths are absolute Path objects. Created via from_config() classmethod.

    This replaces:
    - Bash variables like LOG_DIR, MODEL_DIR at the top of Jinja templates
    - The setup_env() function in worker_setup
    - Scattered path computation throughout the codebase
    """

    # Runtime identifiers
    job_id: str
    run_name: str

    # Node topology
    nodes: Nodes
    head_node_ip: str

    # Computed paths (all absolute)
    log_dir: Path
    results_dir: Path
    model_path: Path
    container_image: Path

    # Resource configuration
    gpus_per_node: int
    network_interface: Optional[str]

    # Container mounts: host_path -> container_path
    container_mounts: Dict[Path, Path] = field(default_factory=dict)

    # Additional srun options
    srun_options: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: "JobConfig",
        job_id: str,
        log_dir_base: Optional[Path] = None,
    ) -> "RuntimeContext":
        """Create RuntimeContext from config and job_id.

        All path computation happens here, once at startup.

        Args:
            config: Validated JobConfig
            job_id: SLURM job ID
            log_dir_base: Base directory for logs (default: ./logs)
        """
        # Get nodes from SLURM
        nodes = Nodes.from_slurm(benchmark_on_separate_node=False)

        # Compute run_name
        run_name = f"{config.name}_{job_id}"

        # Resolve head node IP
        head_node_ip = get_hostname_ip(nodes.head)

        # Compute log directory
        # Format: {log_dir_base}/{job_id}_{prefill_workers}P_{decode_workers}D_{timestamp}
        if log_dir_base is None:
            log_dir_base = Path.cwd() / "logs"

        # Determine worker counts for directory naming
        if config.resources.prefill_workers:
            prefill_workers = config.resources.prefill_workers
            decode_workers = config.resources.decode_workers or 1
            dir_suffix = f"{prefill_workers}P_{decode_workers}D"
        elif config.resources.agg_workers:
            dir_suffix = f"{config.resources.agg_workers}A"
        else:
            dir_suffix = "1P_1D"

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = log_dir_base / f"{job_id}_{dir_suffix}_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Results dir inside log_dir
        results_dir = log_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Resolve model path
        model_path = Path(os.path.expandvars(config.model.path)).resolve()

        # Resolve container image
        container_image = Path(os.path.expandvars(config.model.container)).resolve()

        # Build container mounts
        container_mounts: Dict[Path, Path] = {
            model_path: Path("/model"),
            log_dir: Path("/logs"),
            results_dir: Path("/results"),
        }

        # Add extra mounts from config
        if config.extra_mount:
            for mount_spec in config.extra_mount:
                host_path, container_path = mount_spec.split(":", 1)
                container_mounts[Path(host_path).resolve()] = Path(container_path)

        # Get network interface from cluster config
        from .config import get_srtslurm_setting

        network_interface = get_srtslurm_setting("network_interface", "eth0")

        return cls(
            job_id=job_id,
            run_name=run_name,
            nodes=nodes,
            head_node_ip=head_node_ip,
            log_dir=log_dir,
            results_dir=results_dir,
            model_path=model_path,
            container_image=container_image,
            gpus_per_node=config.resources.gpus_per_node,
            network_interface=network_interface,
            container_mounts=container_mounts,
            srun_options={},
        )

    def get_container_mounts_str(self) -> str:
        """Get container mounts as a comma-separated string for srun."""
        mounts = []
        for host_path, container_path in self.container_mounts.items():
            mounts.append(f"{host_path}:{container_path}")
        return ",".join(mounts)

    def format_string(self, template: str, **extra_kwargs) -> str:
        """Format a template string with runtime values.

        Available placeholders:
            {job_id}, {run_name}, {head_node_ip}, {log_dir}, {results_dir},
            {model_path}, {container_image}, plus any extra_kwargs.
        """
        format_dict = {
            "job_id": self.job_id,
            "run_name": self.run_name,
            "head_node_ip": self.head_node_ip,
            "log_dir": str(self.log_dir),
            "results_dir": str(self.results_dir),
            "model_path": str(self.model_path),
            "container_image": str(self.container_image),
        }
        format_dict.update(extra_kwargs)

        formatted = template.format(**format_dict)
        return os.path.expandvars(formatted)


def get_slurm_job_id() -> Optional[str]:
    """Get the current SLURM job ID from environment."""
    return os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")


def get_slurm_nodelist() -> list[str]:
    """Get list of nodes from SLURM_NODELIST environment variable.

    Returns:
        List of node hostnames, or empty list if not in SLURM.
    """
    import subprocess

    nodelist_raw = os.environ.get("SLURM_NODELIST", "")
    if not nodelist_raw:
        return []

    # Use scontrol to expand the nodelist
    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist_raw],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: try simple parsing for non-ranged formats
        return [nodelist_raw]


def get_hostname_ip(hostname: str, network_interface: Optional[str] = None) -> str:
    """Resolve hostname to IP address.

    Args:
        hostname: Node hostname to resolve
        network_interface: Optional network interface to prefer

    Returns:
        IP address as string
    """
    try:
        # Try socket resolution first
        ip = socket.gethostbyname(hostname)
        return ip
    except socket.gaierror:
        # Fallback: return hostname as-is (may be IP already)
        return hostname

