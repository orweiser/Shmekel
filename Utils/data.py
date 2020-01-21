import numpy as np
import os
import constants


def get_stock_path(base_dir, tckt, ext):
    def strip_tckt(tckt):
        tckt = os.path.split(tckt)[-1]
        if tckt.endswith(ext):
            tckt = tckt[:-(len(ext) + 1)]
        return tckt

    return os.path.join(base_dir, '.'.join((strip_tckt(tckt), ext)))


def load_stock(path, pattern=constants.PATTERN, feature_axis=constants.FEATURE_AXIS):
    log = __file_to_dict(path)

    if not log:
        return log

    a = np.stack([log[key] for key in pattern], axis=feature_axis)

    return a, log['Date']


def __file_to_dict(fname):
    content = _read_a_file(fname)
    if not content:
        return False

    keys = content[0].split(',')
    log = {key: [] for key in keys}
    for line in content[1:]:
        for key, val in zip(keys, line.split(',')):
            if key == 'Date':
                log[key].append(tuple([int(v) for v in val.split('-')]))
            else:
                log[key].append(float(val))
    return log


def _read_a_file(fname):
    with open(fname) as f:
        content = f.read().splitlines(False)
    # you may also want to remove whitespace characters like `\n` at the end of each line
    return content

