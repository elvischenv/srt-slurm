# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pytest: skip-file

import argparse
import json
import sys

"""
A file that parses the response of server health endpoints to check whether the server is ready.

Supports two modes:
- Dynamo frontend: parses /health endpoint JSON with 'instances' key
- SGLang router: parses /workers endpoint JSON with 'stats' key

Usage:

```bash
# Dynamo mode (default)
curl_result=$(curl "${host_ip}:${host_port}/health" 2> /dev/null)
check_result=$(python3 check_server_health.py $N_PREFILL $N_DECODE <<< $curl_result)

# SGLang router mode
curl_result=$(curl "${host_ip}:${host_port}/workers" 2> /dev/null)
check_result=$(python3 check_server_health.py $N_PREFILL $N_DECODE --sglang-router <<< $curl_result)
```
"""


def check_sglang_router_health(expected_n_prefill: int, expected_n_decode: int, response: str) -> str:
    """Check health using sglang router /workers endpoint."""
    try:
        decoded_response = json.loads(response)
    except json.JSONDecodeError:
        return f"Got invalid response from server that leads to JSON Decode error: {response}"

    if "stats" not in decoded_response:
        return f"Key 'stats' not found in response: {response}"

    stats = decoded_response["stats"]
    actual_prefill = stats.get("prefill_count", 0)
    actual_decode = stats.get("decode_count", 0)

    if actual_prefill >= expected_n_prefill and actual_decode >= expected_n_decode:
        return f"Model is ready. Have {actual_prefill} prefills and {actual_decode} decodes."
    else:
        return f"Model is not ready, waiting for {expected_n_prefill - actual_prefill} prefills and {expected_n_decode - actual_decode} decodes. Have {actual_prefill} prefills and {actual_decode} decodes."


def check_dynamo_health(expected_n_prefill: int, expected_n_decode: int, response: str) -> str:
    """Check health using dynamo frontend /health endpoint."""
    try:
        decoded_response = json.loads(response)
    except json.JSONDecodeError:
        return f"Got invalid response from server that leads to JSON Decode error: {response}"

    if "instances" not in decoded_response:
        return f"Key 'instances' not found in response: {response}"

    for instance in decoded_response["instances"]:
        if instance.get("endpoint") == "generate":
            # In disaggregated mode: prefill reports as "prefill", decode reports as "backend"
            # In aggregated mode: workers report as "backend"
            if instance.get("component") == "prefill":
                expected_n_prefill -= 1
            elif instance.get("component") == "decode":
                expected_n_decode -= 1
            elif instance.get("component") == "backend":
                # If we're still waiting for decode workers, count backend as decode
                # Otherwise, count as prefill (aggregated mode)
                if expected_n_decode > 0:
                    expected_n_decode -= 1
                else:
                    expected_n_prefill -= 1

    if expected_n_prefill <= 0 and expected_n_decode <= 0:
        return f"Model is ready. Response: {response}"
    else:
        return f"Model is not ready, waiting for {expected_n_prefill} prefills and {expected_n_decode} decodes to spin up. Response: {response}"


def check_server_health(
    expected_n_prefill: str, expected_n_decode: str, response: str, sglang_router: bool = False
) -> str:
    """
    Checks the health of the server's response and ensures worker counts match expectation.
    """
    if not (expected_n_prefill.isnumeric() and expected_n_decode.isnumeric()):
        return f"Got unparsable expected prefill / decode value: {expected_n_prefill} & {expected_n_decode} should be numeric"

    n_prefill = int(expected_n_prefill)
    n_decode = int(expected_n_decode)

    if sglang_router:
        return check_sglang_router_health(n_prefill, n_decode, response)
    else:
        return check_dynamo_health(n_prefill, n_decode, response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check server health for benchmarking")
    parser.add_argument("n_prefill", help="Expected number of prefill workers")
    parser.add_argument("n_decode", help="Expected number of decode workers")
    parser.add_argument("--sglang-router", action="store_true", help="Use sglang router /workers format")
    args = parser.parse_args()

    response = sys.stdin.read()
    print(
        check_server_health(
            expected_n_prefill=args.n_prefill,
            expected_n_decode=args.n_decode,
            response=response,
            sglang_router=args.sglang_router,
        )
    )
