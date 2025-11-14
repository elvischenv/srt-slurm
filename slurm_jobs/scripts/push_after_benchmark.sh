#!/bin/bash
#
# Helper script to push benchmark results to cloud storage after completion
#
# Usage:
#   ./push_after_benchmark.sh <run_directory>
#
# Environment variables required:
#   AWS_ACCESS_KEY_ID - Cloud storage access key
#   AWS_SECRET_ACCESS_KEY - Cloud storage secret key
#
# Example:
#   export AWS_ACCESS_KEY_ID="your-key"
#   export AWS_SECRET_ACCESS_KEY="your-secret"
#   ./push_after_benchmark.sh 3667_1P_1D_20251110_192145
#

set -e  # Exit on error

# Check if run directory is provided
if [ $# -eq 0 ]; then
    echo "Error: No run directory specified"
    echo "Usage: $0 <run_directory>"
    exit 1
fi

RUN_DIR="$1"

# Check if directory exists
if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Directory '$RUN_DIR' does not exist"
    exit 1
fi

# Check if credentials are set
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Error: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set"
    echo "Export these environment variables before running this script"
    exit 1
fi

# Find the sync script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SYNC_SCRIPT="$SCRIPT_DIR/sync_results.py"

if [ ! -f "$SYNC_SCRIPT" ]; then
    echo "Error: sync_results.py not found at $SYNC_SCRIPT"
    exit 1
fi

# Push the run
echo "Pushing $RUN_DIR to cloud storage..."
python3 "$SYNC_SCRIPT" push "$RUN_DIR"

if [ $? -eq 0 ]; then
    echo "✓ Successfully pushed $RUN_DIR to cloud storage"
    exit 0
else
    echo "✗ Failed to push $RUN_DIR to cloud storage"
    exit 1
fi

