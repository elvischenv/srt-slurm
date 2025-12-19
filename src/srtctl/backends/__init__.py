# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backend implementations for different LLM serving frameworks.

Currently supported:
- SGLang (sglang.py)

Future:
- vLLM
- TensorRT-LLM
"""

from .protocol import BackendProtocol
from .sglang import SGLangBackend

__all__ = ["BackendProtocol", "SGLangBackend"]

