# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Content Cop Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import os
import cv2
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import ContentCopAction, ContentCopObservation
from env.environment import ContentModerationEnv
from inference import predict_frame
from env.reward import compute_reward

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)  # go from server/ → content_cop/

_SHARED_ENV = ContentModerationEnv()
_SHARED_STATE = State(episode_id=str(uuid4()), step_count=0)


class ContentCopEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = ContentCopEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Content Cop environment ready!"
        >>>
        >>> obs = env.step(ContentCopAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # def __init__(self):
    #     self._state = State(episode_id=str(uuid4()), step_count=0)
    #     self._reset_count = 0

    #     self.data = []
    #     self.labels = []

    #     base_path = os.path.join(BASE_DIR, "data")

    #     print("Total files loaded:", len(self.data))

    #     for label, folder in enumerate(["safe", "nsfw", "violence"]):
    #         folder_path = os.path.join(base_path, folder)

    #         for file in os.listdir(folder_path):
    #             self.data.append(os.path.join(folder_path, file))
    #             self.labels.append(label)

    #     self.env = ContentModerationEnv(self.data, self.labels)

    def __init__(self):
        # self.env = ContentModerationEnv()
        self.env = _SHARED_ENV
        # self._state = State(episode_id=str(uuid4()), step_count=0)
        self._global_state = _SHARED_STATE
        self._reset_count = 0

    def reset(self) -> ContentCopObservation:
        # self._state = State(episode_id=str(uuid4()), step_count=0)
        self._global_state.episode_id = str(uuid4())
        self._global_state.step_count = 0
        observation = self.env.reset()
        return observation

    def step(self, action: ContentCopAction):
        observation, reward, done, info = self.env.step(action)
        observation.reward = reward
        observation.done = done
        observation.info = info

        self._global_state.step_count += 1

        return observation

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._global_state
