# Legacy Templates

These are the original Jinja templates that were used before the Python-first
refactoring. They are preserved here for reference and fallback.

## Files

- `job_script_template_disagg.j2` - Disaggregated (prefill/decode) job template
- `job_script_template_agg.j2` - Aggregated job template

## Status

These templates are still used by the legacy submission mode:
```bash
srtctl apply -f config.yaml  # Uses legacy templates
```

The new orchestrator mode uses the minimal template:
```bash
srtctl apply -f config.yaml --use-orchestrator  # Uses job_script_minimal.j2
```

## Migration

The logic from these templates has been moved to Python:

| Template Section | New Location |
|-----------------|--------------|
| Node assignment loops | `srtctl/core/endpoints.py` |
| srun calls | `srtctl/core/utils.py:start_srun_process()` |
| Worker setup | `srtctl/backends/sglang.py` |
| Orchestration | `srtctl/cli/do_sweep.py` |

