#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# IP address resolution utilities for SLURM environments.
# Provides robust IP discovery across different cluster configurations (GB200, H100, etc.)

# Core IP resolution logic - tries multiple methods
# Usage: _resolve_ip "network_interface"
# Returns: IP address on stdout, exits with code 1 on failure
_resolve_ip() {
    local network_interface=$1

    # Method 1: Use specific interface if provided
    if [ -n "$network_interface" ]; then
        ip=$(ip addr show $network_interface 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1)
        if [ -n "$ip" ]; then
            echo "$ip"
            return 0
        fi
    fi

    # Method 2: Use hostname -I (gets first non-loopback IP)
    ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [ -n "$ip" ]; then
        echo "$ip"
        return 0
    fi

    # Method 3: Use ip route to find default source IP
    ip=$(ip route get 8.8.8.8 2>/dev/null | awk -F'src ' 'NR==1{split($2,a," ");print a[1]}')
    if [ -n "$ip" ]; then
        echo "$ip"
        return 0
    fi

    return 1
}

# Get local IP address
# Usage: get_local_ip "network_interface"
# Returns: IP address on stdout, or "127.0.0.1" if all methods fail
get_local_ip() {
    local network_interface=$1
    
    local result
    result=$(_resolve_ip "$network_interface")
    
    if [ -n "$result" ]; then
        echo "$result"
    else
        echo "127.0.0.1"
    fi
}

# Get IP address of a remote SLURM node via srun
# Usage: get_node_ip "node_name" "slurm_job_id" "network_interface"
# Returns: IP address on stdout, exits with code 1 on failure
get_node_ip() {
    local node=$1
    local slurm_job_id=$2
    local network_interface=$3

    # Create inline script with the resolution logic
    local ip_script="
        # Method 1: Use specific interface if provided
        if [ -n \"$network_interface\" ]; then
            ip=\$(ip addr show $network_interface 2>/dev/null | grep 'inet ' | awk '{print \$2}' | cut -d'/' -f1)
            if [ -n \"\$ip\" ]; then
                echo \"\$ip\"
                exit 0
            fi
        fi

        # Method 2: Use hostname -I (gets first non-loopback IP)
        ip=\$(hostname -I 2>/dev/null | awk '{print \$1}')
        if [ -n \"\$ip\" ]; then
            echo \"\$ip\"
            exit 0
        fi

        # Method 3: Use ip route to find default source IP
        ip=\$(ip route get 8.8.8.8 2>/dev/null | awk -F'src ' 'NR==1{split(\$2,a,\" \");print a[1]}')
        if [ -n \"\$ip\" ]; then
            echo \"\$ip\"
            exit 0
        fi

        exit 1
    "

    # Execute the script on target node with single srun command
    local result
    result=$(srun --jobid $slurm_job_id --nodes=1 --ntasks=1 --nodelist=$node bash -c "$ip_script" 2>&1)
    local rc=$?

    if [ $rc -eq 0 ] && [ -n "$result" ]; then
        echo "$result"
        return 0
    else
        echo "Error: Could not retrieve IP address for node $node" >&2
        return 1
    fi
}
