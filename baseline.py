import cv2
from inference import predict_frame
from tasks import task_easy, task_medium, task_hard
from env.grader import grade


def run_task(task_fn):
    data, labels = task_fn()
    predictions = []

    for path in data:
        frame = cv2.imread(path)
        v, n = predict_frame(frame)

        if n > 0.9:
            pred = 2
        elif v > 0.7:
            pred = 1
        else:
            pred = 0

        predictions.append(pred)

    return grade(predictions, labels)


if __name__ == "__main__":
    print("Easy:", run_task(task_easy))
    print("Medium:", run_task(task_medium))
    print("Hard:", run_task(task_hard))
