# CLI Reference

`srtctl` is the main command-line interface for submitting benchmark jobs to SLURM.

## Quick Start

```bash
# Interactive mode - browse recipes, preview, and submit
srtctl

# Submit a job directly
srtctl apply -f recipies/gb200-fp8/sglang-1p4d.yaml

# Preview without submitting
srtctl dry-run -f config.yaml
```

## Interactive Mode

Running `srtctl` with no arguments launches interactive mode:

```bash
srtctl
```

Interactive mode provides:

- **Recipe browser** - Browse and select from `recipies/` folder, organized by subdirectory
- **Config preview** - See model, resources, and benchmark settings at a glance
- **sbatch preview** - View the generated SLURM script before submitting
- **Parameter modification** - Tweak settings before submission
- **Dry-run** - Test without submitting
- **Confirmation** - Review sweep expansion before submitting multiple jobs

### Interactive Menu

After selecting a recipe, you'll see:

```
🚀 Submit job(s)        - Submit to SLURM
👁️  Preview sbatch script - View generated script
✏️  Modify parameters    - Change settings interactively
🔍 Dry-run              - Preview without submitting
📁 Select different config
❌ Exit
```

## Commands

### `srtctl apply`

Submit a job or sweep to SLURM.

```bash
srtctl apply -f <config.yaml> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `-f, --file` | Path to YAML config file (required) |
| `--sweep` | Force sweep mode (usually auto-detected) |
| `--setup-script` | Custom setup script from `configs/` |
| `--tags` | Comma-separated tags for the run |
| `-y, --yes` | Skip confirmation prompts |

**Examples:**

```bash
# Submit single job
srtctl apply -f recipies/gb200-fp8/sglang-1p4d.yaml

# Submit sweep (auto-detected from sweep: section)
srtctl apply -f configs/my-sweep.yaml

# With tags
srtctl apply -f config.yaml --tags "experiment-1,baseline"
```

### `srtctl dry-run`

Preview what would be submitted without actually submitting.

```bash
srtctl dry-run -f <config.yaml> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `-f, --file` | Path to YAML config file (required) |
| `--sweep` | Force sweep mode |

**Examples:**

```bash
# Preview single job - shows sbatch script
srtctl dry-run -f config.yaml

# Preview sweep - shows job table and saves configs
srtctl dry-run -f sweep-config.yaml
```

Dry-run output includes:
- Syntax-highlighted sbatch script
- For sweeps: table of all jobs with parameters
- Generated configs saved to `dry-runs/` folder

## Output

When you submit a job, `srtctl` creates an output directory:

```
outputs/<job_id>/
├── config.yaml         # Copy of submitted config
├── sbatch_script.sh    # Generated SLURM script
└── <job_id>.json       # Job metadata
```

## Sweep Support

Configs with a `sweep:` section are automatically detected and expanded:

```yaml
sweep:
  chunked_prefill_size: [4096, 8192]
  max_total_tokens: [8192, 16384]
```

This creates 4 jobs (2 × 2 Cartesian product). See [Parameter Sweeps](sweeps.md) for details.

## Tips

- Use `srtctl` (no args) for exploring recipes interactively
- Use `srtctl apply -f` for scripting and CI pipelines
- Always `dry-run` first for sweeps to check job count
- Check `outputs/<job_id>/` for submitted configs and metadata

