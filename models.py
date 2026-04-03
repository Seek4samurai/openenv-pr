# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Content Cop Environment.

The content_cop environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel
from typing import Dict, Any


class ContentCopAction(Action):
    """Agent chooses classification"""

    label: int


class ContentCopObservation(BaseModel):
    frame_path: str
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}
