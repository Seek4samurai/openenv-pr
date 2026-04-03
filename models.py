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
from pydantic import Field


class ContentCopAction(Action):
    """Agent chooses classification"""

    label: int


class ContentCopObservation(Observation):
    """Environment gives a frame to classify"""

    frame_path: str
