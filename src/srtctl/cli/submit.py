#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified job submission interface for srtctl.

This is the main entrypoint for submitting benchmarks via YAML configs.

Usage:
    srtctl config.yaml
    srtctl config.yaml --dry-run
    srtctl sweep.yaml --sweep
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import yaml
from datetime import datetime
from pathlib import Path

# Import from srtctl modules
from srtctl.core.config import load_config
from srtctl.core.sweep import generate_sweep_configs
from srtctl.core.backend import SGLangBackend


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def render_commands_file(backend, sglang_config_path: Path, output_path: Path) -> Path:
    """Generate commands.sh with rendered SGLang commands."""
    content = f"""#!/bin/bash
# Generated SGLang commands - Config: {sglang_config_path}

# PREFILL
{backend.render_command(mode="prefill", config_path=sglang_config_path)}

# DECODE
{backend.render_command(mode="decode", config_path=sglang_config_path)}
"""
    output_path.write_text(content)
    output_path.chmod(0o755)
    return output_path


def run_dry_run(config: dict, backend, sglang_config_path: Path = None) -> Path:
    """Execute dry-run: save artifacts and print summary."""
    job_name = config.get("name", "dry-run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.cwd() / "dry-runs" / f"{job_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Save sglang config if present
    has_sglang = False
    if sglang_config_path and sglang_config_path.exists():
        shutil.copy(sglang_config_path, output_dir / "sglang_config.yaml")
        render_commands_file(backend, sglang_config_path, output_dir / "commands.sh")
        has_sglang = True

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({"job_name": job_name, "timestamp": timestamp, "mode": "dry-run"}, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}\n🔍 DRY-RUN: {job_name}\n{'=' * 60}")
    print(f"Output: {output_dir}")
    print(f"Files: config.yaml{', sglang_config.yaml, commands.sh' if has_sglang else ''}, metadata.json")
    print(f"{'=' * 60}\n")

    return output_dir


def submit_single(
    config_path: Path = None,
    config: dict = None,
    dry_run: bool = False,
    setup_script: str = None,
    tags: list[str] = None,
):
    """
    Submit a single job from YAML config.

    Args:
        config_path: Path to YAML config file (or None if config provided)
        config: Pre-loaded config dict (or None if loading from path)
        dry_run: If True, don't submit to SLURM, just validate and save artifacts
        setup_script: Optional custom setup script name in configs directory
        tags: Optional list of tags to apply to the run
    """
    # Load config if needed
    if config is None:
        config = load_config(config_path)
    # else: config already validated and expanded from sweep

    # Dry-run mode
    if dry_run:
        logging.info(f"🔍 DRY-RUN MODE: {config['name']}")
        backend_type = config.get("backend", {}).get("type")
        backend = SGLangBackend(config, setup_script=setup_script) if backend_type == "sglang" else None
        sglang_config_path = backend.generate_config_file() if backend else None
        run_dry_run(config, backend, sglang_config_path)
        return

    # Real submission mode
    logging.info(f"🚀 Submitting job: {config['name']}")

    # Create backend and generate config
    backend_type = config.get("backend", {}).get("type")
    if backend_type == "sglang":
        backend = SGLangBackend(config, setup_script=setup_script)
        sglang_config_path = backend.generate_config_file()

        # Generate SLURM job script using backend
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path, rendered_script = backend.generate_slurm_script(
            config_path=sglang_config_path, timestamp=timestamp
        )

        # Submit to SLURM
        try:
            result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True, check=True)

            # Parse job ID from sbatch output
            job_id = result.stdout.strip().split()[-1]
            logging.info(f"✅ Job submitted successfully with ID: {job_id}")

            # Create log directory
            is_aggregated = config.get("resources", {}).get("prefill_nodes") is None
            if is_aggregated:
                agg_workers = config["resources"]["agg_workers"]
                log_dir_name = f"{job_id}_{agg_workers}A_{timestamp}"
            else:
                prefill_workers = config["resources"]["prefill_workers"]
                decode_workers = config["resources"]["decode_workers"]
                log_dir_name = f"{job_id}_{prefill_workers}P_{decode_workers}D_{timestamp}"

            # Create log directory in srtctl repo
            from srtctl.core.config import get_srtslurm_setting

            srtctl_root_setting = get_srtslurm_setting("srtctl_root")
            if srtctl_root_setting:
                srtctl_root = Path(srtctl_root_setting)
            else:
                # Fall back to current yaml-config directory
                yaml_config_root = Path(__file__).parent.parent.parent.parent
                srtctl_root = yaml_config_root

            log_dir = srtctl_root / "logs" / log_dir_name
            log_dir.mkdir(parents=True, exist_ok=True)

            # Save rendered script
            with open(log_dir / "sbatch_script.sh", "w") as f:
                f.write(rendered_script)

            # Save config
            with open(log_dir / "config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            # Save SGLang config if present
            if sglang_config_path:
                shutil.copy(sglang_config_path, log_dir / "sglang_config.yaml")

            # Generate jobid.json metadata
            resources, model, slurm_cfg = config.get("resources", {}), config.get("model", {}), config.get("slurm", {})
            benchmark_cfg = config.get("benchmark", {})

            run_meta = {
                "slurm_job_id": job_id,
                "run_date": timestamp,
                "job_name": config.get("name", "unnamed"),
                "account": slurm_cfg.get("account"),
                "partition": slurm_cfg.get("partition"),
                "time_limit": slurm_cfg.get("time_limit"),
                "container": model.get("container"),
                "model_dir": model.get("path"),
                "gpus_per_node": resources.get("gpus_per_node"),
                "gpu_type": config.get("backend", {}).get("gpu_type"),
                "mode": "aggregated" if is_aggregated else "disaggregated",
            }
            if is_aggregated:
                run_meta.update(agg_nodes=resources.get("agg_nodes"), agg_workers=resources.get("agg_workers"))
            else:
                run_meta.update(
                    prefill_nodes=resources.get("prefill_nodes"),
                    decode_nodes=resources.get("decode_nodes"),
                    prefill_workers=resources.get("prefill_workers"),
                    decode_workers=resources.get("decode_workers"),
                )

            metadata = {
                "version": "1.0",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_metadata": run_meta,
            }

            if bench_type := benchmark_cfg.get("type", "manual"):
                bench_meta = {"type": bench_type}
                if bench_type == "sa-bench":
                    conc = benchmark_cfg.get("concurrencies", [])
                    bench_meta.update(
                        isl=str(benchmark_cfg.get("isl", "")),
                        osl=str(benchmark_cfg.get("osl", "")),
                        concurrencies="x".join(str(c) for c in conc) if isinstance(conc, list) else str(conc or ""),
                        **{"req-rate": str(benchmark_cfg.get("req_rate", "inf"))},
                    )
                metadata["profiler_metadata"] = bench_meta
            if tags:
                metadata["tags"] = tags

            with open(log_dir / f"{job_id}.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logging.info(f"📁 Logs directory: {log_dir}")
            print(f"\n✅ Job {job_id} submitted!")
            print(f"📁 Logs: {log_dir}\n")

        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Error submitting job: {e}")
            logging.error(f"stderr: {e.stderr}")
            raise
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def is_sweep_config(config_path: Path) -> bool:
    """Check if config file is a sweep config by looking for 'sweep' section."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return "sweep" in config if config else False
    except Exception:
        return False


def submit_sweep(config_path: Path, dry_run: bool = False, setup_script: str = None, tags: list[str] = None):
    """
    Submit parameter sweep.

    Args:
        config_path: Path to sweep YAML config
        dry_run: If True, don't submit to SLURM, just validate and save artifacts
        setup_script: Optional custom setup script name in configs directory
        tags: Optional list of tags to apply to all runs in the sweep
    """
    # Load YAML directly without validation (sweep configs have extra 'sweep' field)
    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)

    # Generate all configs
    configs = generate_sweep_configs(sweep_config)
    logging.info(f"Generated {len(configs)} configurations for sweep")

    if dry_run:
        logging.info(f"🔍 DRY-RUN MODE: Sweep with {len(configs)} jobs")

        # Create sweep output directory
        sweep_dir = Path.cwd() / "dry-runs" / f"{sweep_config['name']}_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"📁 Sweep directory: {sweep_dir}")

        # Save sweep config
        with open(sweep_dir / "sweep_config.yaml", "w") as f:
            yaml.dump(sweep_config, f, default_flow_style=False)

        # Generate each job
        for i, (config, params) in enumerate(configs, 1):
            logging.info(f"\n[{i}/{len(configs)}] {config['name']}")
            logging.info(f"  Parameters: {params}")

            # Create job directory
            job_dir = sweep_dir / f"job_{i:03d}_{config['name']}"
            job_dir.mkdir(exist_ok=True)

            # Save config
            with open(job_dir / "config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            # Generate SGLang config and commands
            if config.get("backend", {}).get("type") == "sglang":
                backend = SGLangBackend(config, setup_script=setup_script)
                sglang_config_path = backend.generate_config_file(params)
                if sglang_config_path:
                    shutil.copy(sglang_config_path, job_dir / "sglang_config.yaml")

                    # Save rendered commands (like single dry-run does)
                    render_commands_file(backend, sglang_config_path, job_dir / "commands.sh")

            logging.info(f"  ✓ {job_dir.name}")

        print(
            f"\n{'=' * 60}\n🔍 SWEEP: {sweep_config['name']} ({len(configs)} jobs)\nOutput: {sweep_dir}\n{'=' * 60}\n"
        )
        return

    # Real submission
    for i, (config, params) in enumerate(configs, 1):
        logging.info(f"\n[{i}/{len(configs)}] Submitting: {config['name']}")
        logging.info(f"  Parameters: {params}")
        submit_single(config=config, dry_run=False, setup_script=setup_script, tags=tags)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="srtctl - SLURM job submission",
        epilog="Examples:\n  srtctl apply -f config.yaml\n  srtctl dry-run -f sweep.yaml --sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common args for both commands
    def add_common_args(p):
        p.add_argument("-f", "--file", type=Path, required=True, dest="config", help="YAML config file")
        p.add_argument("--sweep", action="store_true", help="Force sweep mode")

    apply_parser = subparsers.add_parser("apply", help="Submit job(s) to SLURM")
    add_common_args(apply_parser)
    apply_parser.add_argument("--setup-script", type=str, help="Custom setup script in configs/")
    apply_parser.add_argument("--tags", type=str, help="Comma-separated tags")

    dry_run_parser = subparsers.add_parser("dry-run", help="Validate without submitting")
    add_common_args(dry_run_parser)

    args = parser.parse_args()
    if not args.config.exists():
        logging.error(f"Config not found: {args.config}")
        sys.exit(1)

    is_dry_run = args.command == "dry-run"
    is_sweep = args.sweep or is_sweep_config(args.config)
    tags = [t.strip() for t in (getattr(args, "tags", "") or "").split(",") if t.strip()] or None

    try:
        setup_script = getattr(args, "setup_script", None)
        if is_sweep:
            submit_sweep(args.config, dry_run=is_dry_run, setup_script=setup_script, tags=tags)
        else:
            submit_single(config_path=args.config, dry_run=is_dry_run, setup_script=setup_script, tags=tags)
    except Exception as e:
        logging.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
