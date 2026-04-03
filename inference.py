from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


violence_model = load_model("./checkpoints/gore/violence_best_model.h5")
nsfw_model = load_model("./checkpoints/nsfw/v2_nsfw_model.h5")


def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def predict_frame(frame):
    img = preprocess(frame)

    v = float(violence_model.predict(img, verbose=0)[0][0])
    n = float(nsfw_model.predict(img, verbose=0)[0][0])

    return v, n
