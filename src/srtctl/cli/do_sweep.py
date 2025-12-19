# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main orchestration script for benchmark sweeps.

This script is called from within the sbatch job and coordinates:
1. Starting head node infrastructure (NATS, etcd)
2. Starting backend workers (prefill/decode/agg)
3. Starting frontends and nginx
4. Running benchmarks
5. Cleanup

This replaces the complex bash logic in job_script_template_disagg.j2.
"""

import argparse
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from srtctl.backends.sglang import SGLangBackend, build_sglang_command_from_config
from srtctl.core.config import load_config
from srtctl.core.endpoints import Endpoint, Process, allocate_endpoints, endpoints_to_processes
from srtctl.core.process_registry import (
    ManagedProcess,
    NamedProcesses,
    ProcessRegistry,
    setup_signal_handlers,
    start_process_monitor,
)
from srtctl.core.runtime import RuntimeContext, get_hostname_ip, get_slurm_job_id
from srtctl.core.utils import start_srun_process, wait_for_health, wait_for_port

logger = logging.getLogger(__name__)

# Emoji constants for logging
ROCKET = "🚀"
CHECK = "✓"
CROSS = "✗"
GEAR = "⚙"
HOURGLASS = "⏳"
PACKAGE = "📦"
WRENCH = "🔧"


def section(title: str, emoji: str = GEAR) -> None:
    """Print a section header."""
    logger.info("")
    logger.info("%s %s", emoji, title)
    logger.info("-" * 60)


@dataclass
class SweepOrchestrator:
    """Main orchestrator for benchmark sweeps.

    This class coordinates all the moving parts:
    - Infrastructure (NATS, etcd)
    - Backend workers (SGLang prefill/decode/agg)
    - Frontends (nginx, dynamo frontend or sglang-router)
    - Benchmarks

    Usage:
        config = load_config(config_path)
        runtime = RuntimeContext.from_config(config, job_id)
        orchestrator = SweepOrchestrator(config, runtime)
        exit_code = orchestrator.run()
    """

    config: Dict
    runtime: RuntimeContext

    def __post_init__(self):
        """Initialize derived attributes."""
        self._backend = SGLangBackend.from_job_config(self.config)
        self._endpoints: Optional[List[Endpoint]] = None
        self._processes: Optional[List[Process]] = None

    @property
    def endpoints(self) -> List[Endpoint]:
        """Lazily compute endpoint allocation."""
        if self._endpoints is None:
            resources = self.config["resources"]

            # Determine worker counts
            num_prefill = resources.get("prefill_workers", 0) or 0
            num_decode = resources.get("decode_workers", 0) or 0
            num_agg = resources.get("agg_workers", 0) or 0

            # Determine GPUs per worker
            prefill_nodes = resources.get("prefill_nodes", 0) or 0
            decode_nodes = resources.get("decode_nodes", 0) or 0
            agg_nodes = resources.get("agg_nodes", 0) or 0
            gpus_per_node = resources.get("gpus_per_node", 8)

            # Calculate GPUs per worker
            gpus_per_prefill = (
                (prefill_nodes * gpus_per_node) // num_prefill
                if num_prefill > 0
                else gpus_per_node
            )
            gpus_per_decode = (
                (decode_nodes * gpus_per_node) // num_decode
                if num_decode > 0
                else gpus_per_node
            )
            gpus_per_agg = (
                (agg_nodes * gpus_per_node) // num_agg
                if num_agg > 0
                else gpus_per_node
            )

            self._endpoints = allocate_endpoints(
                num_prefill=num_prefill,
                num_decode=num_decode,
                num_agg=num_agg,
                gpus_per_prefill=gpus_per_prefill,
                gpus_per_decode=gpus_per_decode,
                gpus_per_agg=gpus_per_agg,
                gpus_per_node=gpus_per_node,
                available_nodes=self.runtime.nodes.worker,
            )

        return self._endpoints

    @property
    def backend_processes(self) -> List[Process]:
        """Lazily compute physical processes from endpoints."""
        if self._processes is None:
            self._processes = endpoints_to_processes(self.endpoints)
        return self._processes

    def start_head_infrastructure(
        self,
        registry: ProcessRegistry,
    ) -> ManagedProcess:
        """Start NATS and etcd on the head node.

        Args:
            registry: Process registry for tracking

        Returns:
            ManagedProcess for the infrastructure
        """
        section("Starting head node infrastructure", ROCKET)
        logger.info("Head node: %s", self.runtime.nodes.head)

        # Find the setup_head.py script
        setup_script = Path(__file__).parent / "setup_head.py"
        if not setup_script.exists():
            raise RuntimeError(f"setup_head.py not found at {setup_script}")

        # Mount it into container
        setup_script_container = Path("/tmp/setup_head.py")

        infra_log = self.runtime.log_dir / f"infrastructure_{self.runtime.job_id}.log"

        # Build command
        cmd = [
            "python3",
            str(setup_script_container),
            "--name",
            self.config["name"],
            "--log-dir",
            str(self.runtime.log_dir),
        ]

        # Build mounts
        mounts = dict(self.runtime.container_mounts)
        mounts[setup_script] = setup_script_container

        proc = start_srun_process(
            command=cmd,
            nodelist=[self.runtime.nodes.head],
            output=str(infra_log),
            container_image=str(self.runtime.container_image),
            container_mounts=mounts,
        )

        managed = ManagedProcess(
            name="head_infrastructure",
            popen=proc,
            log_file=infra_log,
            node=self.runtime.nodes.head,
            critical=True,
        )

        # Wait for NATS and etcd
        logger.info("Waiting for NATS (port 4222)...")
        if not wait_for_port(self.runtime.nodes.head, 4222, timeout=60):
            raise RuntimeError("NATS failed to start")
        logger.info("%s NATS is ready", CHECK)

        logger.info("Waiting for etcd (port 2379)...")
        if not wait_for_port(self.runtime.nodes.head, 2379, timeout=60):
            raise RuntimeError("etcd failed to start")
        logger.info("%s etcd is ready", CHECK)

        return managed

    def start_worker(
        self,
        process: Process,
        endpoint_processes: List[Process],
    ) -> ManagedProcess:
        """Start a single worker process.

        Args:
            process: The Process to start
            endpoint_processes: All processes in this endpoint

        Returns:
            ManagedProcess for tracking
        """
        mode = process.endpoint_mode
        index = process.endpoint_index

        section(
            f"Starting {mode} worker {index} on {process.node}",
            WRENCH,
        )

        # Get config for this mode
        config = self._backend.get_config_for_mode(mode)

        # Get leader IP for distributed init
        endpoint_nodes = [p.node for p in endpoint_processes]
        leader_ip = get_hostname_ip(endpoint_nodes[0])

        # Build command
        cmd = build_sglang_command_from_config(
            mode=mode,
            config=config,
            model_path=self.runtime.model_path,
            served_model_name=self._backend.served_model_name,
            leader_ip=leader_ip,
            dist_init_port=29500,
            num_nodes=len(endpoint_nodes),
            node_rank=process.node_rank,
            use_sglang_router=self._backend.use_sglang_router,
            dump_config_path=self.runtime.log_dir
            / f"{mode}_config_{index}_{process.node}.json",
        )

        worker_log = (
            self.runtime.log_dir / f"{mode}_{index}_{process.node}_{self.runtime.job_id}.log"
        )

        # Environment variables
        env_to_set = {
            "HEAD_NODE_IP": self.runtime.head_node_ip,
            "ETCD_ENDPOINTS": f"http://{self.runtime.nodes.head}:2379",
            "NATS_SERVER": f"nats://{self.runtime.nodes.head}:4222",
            "DYN_SYSTEM_PORT": str(process.sys_port),
        }

        # Set CUDA_VISIBLE_DEVICES if not using all GPUs
        if len(process.gpu_indices) < self.runtime.gpus_per_node:
            env_to_set["CUDA_VISIBLE_DEVICES"] = process.cuda_visible_devices

        logger.info("Command: %s", shlex.join(cmd))
        logger.info("Log: %s", worker_log)

        proc = start_srun_process(
            command=cmd,
            nodelist=[process.node],
            output=str(worker_log),
            container_image=str(self.runtime.container_image),
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
        )

        return ManagedProcess(
            name=f"{mode}_{index}_{process.node}",
            popen=proc,
            log_file=worker_log,
            node=process.node,
            critical=True,
        )

    def start_all_workers(self) -> NamedProcesses:
        """Start all backend workers.

        Returns:
            Dict mapping names to ManagedProcess objects
        """
        section("Starting backend workers", PACKAGE)

        # Group processes by endpoint
        from collections import defaultdict

        grouped: Dict[tuple, List[Process]] = defaultdict(list)
        for process in self.backend_processes:
            key = (process.endpoint_mode, process.endpoint_index)
            grouped[key].append(process)

        result: NamedProcesses = {}

        for endpoint_key, endpoint_processes in grouped.items():
            for process in endpoint_processes:
                managed = self.start_worker(process, endpoint_processes)
                result[managed.name] = managed

        logger.info("Started %d worker processes", len(result))
        return result

    def start_frontend(self, registry: ProcessRegistry) -> Optional[ManagedProcess]:
        """Start the frontend (nginx + dynamo frontend or sglang-router).

        For simplicity, we start a single frontend on the head node.
        Multi-frontend support can be added later.

        Args:
            registry: Process registry

        Returns:
            ManagedProcess for the frontend, or None if using sglang-router
        """
        section("Starting frontend", PACKAGE)

        frontend_config = self.config.get("frontend", {})
        use_sglang_router = frontend_config.get("use_sglang_router", False)

        if use_sglang_router:
            # Start sglang-router
            return self._start_sglang_router(registry)
        else:
            # Start dynamo frontend
            return self._start_dynamo_frontend(registry)

    def _start_dynamo_frontend(self, registry: ProcessRegistry) -> ManagedProcess:
        """Start dynamo frontend on the head node."""
        logger.info("Starting dynamo frontend on %s", self.runtime.nodes.head)

        frontend_log = self.runtime.log_dir / f"frontend_{self.runtime.job_id}.log"

        # Get extra args from config
        frontend_config = self.config.get("frontend", {})
        extra_args = frontend_config.get("dynamo_frontend_args", {})

        cmd = ["python3", "-m", "dynamo.frontend", "--http-port=8000"]

        # Add extra args
        for key, value in (extra_args or {}).items():
            if value is True:
                cmd.append(f"--{key}")
            elif value is not False and value is not None:
                cmd.extend([f"--{key}", str(value)])

        env_to_set = {
            "ETCD_ENDPOINTS": f"http://{self.runtime.nodes.head}:2379",
            "NATS_SERVER": f"nats://{self.runtime.nodes.head}:4222",
        }

        proc = start_srun_process(
            command=cmd,
            nodelist=[self.runtime.nodes.head],
            output=str(frontend_log),
            container_image=str(self.runtime.container_image),
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
        )

        return ManagedProcess(
            name="frontend",
            popen=proc,
            log_file=frontend_log,
            node=self.runtime.nodes.head,
            critical=True,
        )

    def _start_sglang_router(self, registry: ProcessRegistry) -> ManagedProcess:
        """Start sglang-router on the head node."""
        logger.info("Starting sglang-router on %s", self.runtime.nodes.head)

        router_log = self.runtime.log_dir / f"router_{self.runtime.job_id}.log"

        # Collect prefill and decode leader IPs
        prefill_ips = []
        decode_ips = []

        for endpoint in self.endpoints:
            leader_ip = get_hostname_ip(endpoint.leader_node)
            if endpoint.mode == "prefill":
                prefill_ips.append(leader_ip)
            elif endpoint.mode == "decode":
                decode_ips.append(leader_ip)

        # Build router command
        cmd = ["python", "-m", "sglang_router.launch_router", "--pd-disaggregation"]

        server_port = 30000
        bootstrap_port = 30001

        for ip in prefill_ips:
            cmd.extend(["--prefill", f"http://{ip}:{server_port}", str(bootstrap_port)])

        for ip in decode_ips:
            cmd.extend(["--decode", f"http://{ip}:{server_port}"])

        cmd.extend(["--host", "0.0.0.0", "--port", "8000"])

        # Add extra args from config
        frontend_config = self.config.get("frontend", {})
        extra_args = frontend_config.get("sglang_router_args", {})
        for key, value in (extra_args or {}).items():
            if value is True:
                cmd.append(f"--{key}")
            elif value is not False and value is not None:
                cmd.extend([f"--{key}", str(value)])

        logger.info("Router command: %s", shlex.join(cmd))

        proc = start_srun_process(
            command=cmd,
            nodelist=[self.runtime.nodes.head],
            output=str(router_log),
            container_image=str(self.runtime.container_image),
            container_mounts=self.runtime.container_mounts,
        )

        return ManagedProcess(
            name="sglang_router",
            popen=proc,
            log_file=router_log,
            node=self.runtime.nodes.head,
            critical=True,
        )

    def run_benchmark(
        self,
        registry: ProcessRegistry,
        stop_event: threading.Event,
    ) -> int:
        """Run the benchmark.

        Args:
            registry: Process registry for failure detection
            stop_event: Event to signal abort

        Returns:
            Exit code (0 for success)
        """
        section("Running benchmark", ROCKET)

        # Wait for server to be healthy
        resources = self.config["resources"]
        num_workers = (
            (resources.get("prefill_workers", 0) or 0)
            + (resources.get("decode_workers", 0) or 0)
            + (resources.get("agg_workers", 0) or 0)
        )

        logger.info("Waiting for server health (expecting %d workers)...", num_workers)

        if not wait_for_health(
            self.runtime.nodes.head,
            8000,
            max_attempts=60,
            interval=10.0,
            expected_workers=num_workers,
            stop_event=stop_event,
        ):
            logger.error("%s Server did not become healthy", CROSS)
            return 1

        logger.info("%s Server is healthy", CHECK)

        # Get benchmark config
        benchmark_config = self.config.get("benchmark", {})
        benchmark_type = benchmark_config.get("type", "manual")

        if benchmark_type == "manual":
            logger.info("Benchmark type is 'manual' - server is ready for testing")
            logger.info("Frontend URL: http://%s:8000", self.runtime.nodes.head)
            logger.info("Press Ctrl+C to stop the job")

            # Wait forever (or until signal)
            while not stop_event.is_set():
                if registry.check_failures():
                    logger.error("Worker failure detected during manual mode")
                    return 1
                time.sleep(5)

            return 0

        # For other benchmark types, run the benchmark script
        # This is a simplified version - expand as needed
        logger.info("Benchmark type '%s' - running benchmark", benchmark_type)

        # TODO: Implement actual benchmark running
        # For now, just wait a bit and return success
        time.sleep(10)

        return 0

    def run(self) -> int:
        """Run the complete sweep.

        Returns:
            Exit code (0 for success)
        """
        section("Sweep Orchestrator", ROCKET)
        logger.info("Job ID: %s", self.runtime.job_id)
        logger.info("Run name: %s", self.runtime.run_name)
        logger.info("Head node: %s", self.runtime.nodes.head)
        logger.info("Worker nodes: %s", ", ".join(self.runtime.nodes.worker))

        # Setup process registry and signal handlers
        registry = ProcessRegistry(job_id=self.runtime.job_id)
        stop_event = threading.Event()
        setup_signal_handlers(stop_event, registry)
        start_process_monitor(stop_event, registry)

        exit_code = 1

        try:
            # Start infrastructure
            head_proc = self.start_head_infrastructure(registry)
            registry.add_process(head_proc)

            # Start workers
            worker_procs = self.start_all_workers()
            registry.add_processes(worker_procs)

            # Start frontend
            frontend_proc = self.start_frontend(registry)
            if frontend_proc:
                registry.add_process(frontend_proc)

            # Run benchmark
            exit_code = self.run_benchmark(registry, stop_event)

        except Exception as e:
            logger.exception("Error during sweep: %s", e)
            exit_code = 1

        finally:
            section("Cleanup", PACKAGE)
            stop_event.set()
            registry.cleanup()

            if exit_code != 0:
                registry.print_failure_details()

        return exit_code


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run benchmark sweep")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        # Load config
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error("Config file not found: %s", config_path)
            sys.exit(1)

        config = load_config(config_path)

        # Get job ID
        job_id = get_slurm_job_id()
        if not job_id:
            logger.error("Not running in SLURM (SLURM_JOB_ID not set)")
            sys.exit(1)

        # Create runtime context
        runtime = RuntimeContext.from_config(config, job_id)

        # Run orchestrator
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        exit_code = orchestrator.run()

        sys.exit(exit_code)

    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

