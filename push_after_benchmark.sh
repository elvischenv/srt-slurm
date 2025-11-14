#!/bin/bash
#
# Push benchmark results to cloud storage
#
# Usage:
#   ./push_after_benchmark.sh --log-dir <logs_directory>
#   ./push_after_benchmark.sh <single_run_directory>
#
# Examples:
#   ./push_after_benchmark.sh --log-dir /mnt/lustre01/users-public/slurm-shared/joblogs
#   ./push_after_benchmark.sh 3667_1P_1D_20251110_192145
#

set -e  # Exit on error

# Find the sync script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SYNC_SCRIPT="$SCRIPT_DIR/slurm_jobs/scripts/sync_results.py"

if [ ! -f "$SYNC_SCRIPT" ]; then
    echo "Error: sync_results.py not found at $SYNC_SCRIPT"
    exit 1
fi

# Check if credentials are set
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Error: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set"
    echo "Export these environment variables before running this script"
    exit 1
fi

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided"
    echo "Usage: $0 --log-dir <logs_directory>"
    echo "   or: $0 <single_run_directory>"
    exit 1
fi

if [ "$1" = "--log-dir" ]; then
    # Push all runs from logs directory
    if [ $# -lt 2 ]; then
        echo "Error: --log-dir requires a directory path"
        exit 1
    fi
    
    LOGS_DIR="$2"
    
    if [ ! -d "$LOGS_DIR" ]; then
        echo "Error: Directory '$LOGS_DIR' does not exist"
        exit 1
    fi
    
    echo "Pushing all runs from $LOGS_DIR to cloud storage..."
    python3 "$SYNC_SCRIPT" --logs-dir "$LOGS_DIR" push-all
    
else
    # Push single run directory
    RUN_DIR="$1"
    
    if [ ! -d "$RUN_DIR" ]; then
        echo "Error: Directory '$RUN_DIR' does not exist"
        exit 1
    fi
    
    echo "Pushing $RUN_DIR to cloud storage..."
    python3 "$SYNC_SCRIPT" push "$RUN_DIR"
fi

if [ $? -eq 0 ]; then
    echo "✓ Successfully pushed to cloud storage"
    exit 0
else
    echo "✗ Failed to push to cloud storage"
    exit 1
fi

