import os


def get_label_from_path(path):
    if "nsfw" in path:
        return 2
    elif "violence" in path:
        return 1
    else:
        return 0


def load_task_data(folder):
    data = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".jpg"):
                data.append(os.path.join(root, f))
    labels = [get_label_from_path(p) for p in data]
    return data, labels


def task_easy():
    return load_task_data("data/tasks/easy")


def task_medium():
    return load_task_data("data/tasks/medium")


def task_hard():
    return load_task_data("data/tasks/hard")
