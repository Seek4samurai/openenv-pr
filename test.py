from env.environment import ContentModerationEnv
from inference import predict_frame

import cv2


def get_label_from_path(path):
    if "nsfw" in path:
        return 2
    elif "violence" in path:
        return 1
    else:
        return 0


data = ["data/safe/00a2q1yv76h01.jpg", "data/nsfw/0dkrd89af6o31.jpg"]
labels = [get_label_from_path(p) for p in data]

env = ContentModerationEnv(data, labels)

state = env.reset()

done = False

while not done:
    frame = cv2.imread(state.frame_path)

    if frame is None:
        raise ValueError(f"Failed to load: {state}")

    v, n = predict_frame(frame)

    # model decides action
    if n > 0.9:
        action = 2
    elif v > 0.7:
        action = 1
    else:
        action = 0

    state, reward, done, info = env.step(action)

    print("Action:", action)
    print("Info:", info)
    print("Reward:", reward)
    print("-----")
