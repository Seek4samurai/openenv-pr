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

import cv2
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import ContentCopAction, ContentCopObservation
from .inference import predict_frame
from env.reward import compute_reward


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

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        self.data = []
        self.labels = []

        base_path = "data"

        for label, folder in enumerate(["safe", "nsfw", "violence"]):
            folder_path = os.path.join(base_path, folder)

            for file in os.listdir(folder_path):
                self.data.append(os.path.join(folder_path, file))
                self.labels.append(label)

        self.env = ContentModerationEnv(self.data, self.labels)

    def reset(self) -> ContentCopObservation:
        """Reset the environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)

        state = self.env.reset()

        return ContentCopObservation(
            frame_path=self.data[0],
            reward=0.0,
            done=False,
        )

    def step(self, action: ContentCopAction) -> ContentCopObservation:  # type: ignore[override]
        """Execute a step in the environment by echoing the message."""
        frame = cv2.imread(self.current_frame_path)

        v, n = predict_frame(frame)

        # simple ground truth logic (you can improve later)
        if n > 0.8:
            true_label = 2  # NSFW
        elif v > 0.7:
            true_label = 1  # VIOLENCE
        else:
            true_label = 0  # SAFE

        # compare agent vs ground truth
        reward = compute_reward(action, label)

        return observation, reward, done, {}

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
