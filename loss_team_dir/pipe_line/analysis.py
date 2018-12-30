from . import _experiments_dir
import pickle
import os


def get_saved_experiments_names_paths(exp_dir=_experiments_dir):
    exp_names, paths = [[], []]

    if type(exp_dir) is list:
        for d in exp_dir:
            a, b = get_saved_experiments_names_paths(d)
            exp_names += a
            paths += b
        return exp_names, paths

    for item in os.listdir(exp_dir):
        path = os.path.join(exp_dir, item)
        if not os.path.isdir(path):
            continue

        exp_names.append(item)
        paths.append(path)

    return exp_names, paths


def load_exp_history(exp_path):
    history_path = os.path.join(exp_path, 'history.h5')
    with open(history_path, 'rb') as f:
        return pickle.load(f)





