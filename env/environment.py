import cv2
from inference import predict_frame
from models import ContentCopObservation
from .reward import compute_reward


class ContentModerationEnv:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.index = 0

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

        # 🔥 model prediction
        v, n = predict_frame(frame)

        # convert to class label
        if n > 0.7:
            pred = 2  # NSFW
        elif v > 0.6:
            pred = 1  # VIOLENCE
        else:
            pred = 0  # SAFE

        # ✅ actual label
        label = self.labels[self.index]

        reward = compute_reward(action, label)

        info = {"model_pred": pred, "true_label": label}

        self.index += 1
        done = self.index >= len(self.data)

        next_state = None if done else self.data[self.index]
        print("Action:", action)
        print("True label:", label)
        print("Model pred:", pred)

        if done:
            observation = None
        else:
            observation = ContentCopObservation(frame_path=self.data[self.index])

        return observation, reward, done, info
