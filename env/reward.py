def compute_reward(action, label):
    if action == label:
        return 1.0
    elif abs(action - label) == 1:
        return -0.5
    else:
        return -1.0
