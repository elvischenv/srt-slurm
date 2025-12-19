# srtctl - Python-first SLURM Orchestration

This package provides Python-first orchestration for LLM inference benchmarks
on SLURM clusters, replacing the previous Jinja/bash-heavy approach.

## Architecture

```
srtctl/
├── __init__.py              # Package exports
├── cli/
│   ├── submit.py            # Job submission CLI
│   ├── do_sweep.py          # Main orchestrator (called by sbatch)
│   └── setup_head.py        # Head node infrastructure (NATS/etcd)
├── core/
│   ├── config.py            # Config loading and validation
│   ├── runtime.py           # RuntimeContext - single source of truth
│   ├── endpoints.py         # Endpoint/Process dataclasses
│   ├── process_registry.py  # Process lifecycle management
│   ├── utils.py             # srun helpers, wait functions
│   ├── schema.py            # Pydantic schemas
│   ├── sweep.py             # Sweep parameter handling
│   └── backend.py           # Legacy SGLangBackend
└── backends/
    ├── protocol.py          # BackendProtocol interface
    └── sglang.py            # SGLang implementation
```

## Usage

### Legacy Mode (existing behavior)
```bash
srtctl apply -f config.yaml
```

### Orchestrator Mode (new Python-first approach)
```bash
srtctl apply -f config.yaml --use-orchestrator
```

## Key Concepts

### RuntimeContext
Single source of truth for all computed paths and values. Replaces bash
variables scattered throughout Jinja templates.

```python
runtime = RuntimeContext.from_config(config, job_id)
print(runtime.log_dir)       # Computed once
print(runtime.model_path)    # Resolved from config
print(runtime.head_node_ip)  # From SLURM
```

### Endpoints and Processes
Typed Python replaces bash array math:

```python
# Old (Jinja/bash):
# for i in $(seq 0 $((PREFILL_WORKERS - 1))); do
#     leader_idx=$((WORKER_NODE_OFFSET + i * PREFILL_NODES_PER_WORKER))
# done

# New (Python):
endpoints = allocate_endpoints(
    num_prefill=2, num_decode=4,
    gpus_per_prefill=8, gpus_per_decode=4,
    gpus_per_node=8, available_nodes=nodes
)
for endpoint in endpoints:
    print(f"{endpoint.mode} worker {endpoint.index} on {endpoint.nodes}")
```

### ProcessRegistry
Manages process lifecycle with health monitoring:

```python
registry = ProcessRegistry(job_id)
registry.add_process(worker_proc)

# Background thread monitors for failures
if registry.check_failures():
    registry.cleanup()  # Graceful shutdown
```

### BackendProtocol
Interface for different serving frameworks:

```python
class BackendProtocol(Protocol):
    def allocate_endpoints(self, ...) -> List[Endpoint]: ...
    def endpoints_to_processes(self, ...) -> List[Process]: ...
    def start_processes(self, ...) -> NamedProcesses: ...
```

## Migration from Legacy

The legacy Jinja templates are preserved in `scripts/templates.legacy/`.
To use the new orchestrator:

1. Add `--use-orchestrator` flag to `srtctl apply`
2. The minimal sbatch template (`job_script_minimal.j2`) calls `do_sweep.py`
3. All bash logic is replaced by Python in `SweepOrchestrator`

## Files Overview

| New File | Replaces | Purpose |
|----------|----------|---------|
| `core/runtime.py` | Bash vars in Jinja | Computed paths/values |
| `core/endpoints.py` | Bash array math | Worker topology |
| `core/process_registry.py` | fire-and-forget `&` | Process management |
| `cli/do_sweep.py` | 550-line Jinja template | Orchestration |
| `backends/sglang.py` | worker_setup/*.py | SGLang launching |

