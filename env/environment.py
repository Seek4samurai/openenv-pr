import os
import cv2
from ..inference import predict_frame
from ..models import ContentCopObservation
from .reward import compute_reward


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)  # go to content_cop/


class ContentModerationEnv:
    def __init__(self):
        self.data = []
        self.labels = []
        self.index = 0

        print("ENV CREATED")

        base_path = os.path.join(BASE_DIR, "data")

        for label, folder in enumerate(["safe", "nsfw", "violence"]):
            folder_path = os.path.join(base_path, folder)

            for file in os.listdir(folder_path):
                self.data.append(os.path.join(folder_path, file))
                self.labels.append(label)

        print("Loaded samples:", len(self.data))

    def reset(self):
        self.index = 0
        return ContentCopObservation(frame_path=self.data[self.index])

    def state(self):
        return ContentCopObservation(frame_path=self.data[self.index])

    def step(self, action):
        img_path = self.data[self.index]
        frame = cv2.imread(img_path)

        if frame is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # prediction
        v, n = predict_frame(frame)

        if n > 0.7:
            pred = 2
        elif v > 0.6:
            pred = 1
        else:
            pred = 0

        label = self.labels[self.index]
        reward = compute_reward(pred, label)

        info = {"model_pred": pred, "true_label": label}

        # ✅ create observation FIRST (current frame)
        observation = ContentCopObservation(frame_path=img_path)

        # THEN move forward
        self.index += 1
        done = self.index >= len(self.data)

        print("INDEX:", self.index)
        print("True label:", label)
        print("Model pred:", pred)
        print("STEP INDEX:", self.index, "FILE:", img_path)

        return observation, reward, done, info
