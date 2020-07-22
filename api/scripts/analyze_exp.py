import argparse
import os
import sys

sys.path.insert(0, os.path.join('..', '..', '..', __file__))
from api.core import get_exp_from_config, load_config


def main(path, metrics=None):
    config = load_config(path)
    exp = get_exp_from_config(config)

    params = {}
    if metrics:
        params['metrics'] = metrics

    exp.results.plot(**params)
    for m in metrics:
        exp.results.get_best_epoch(m).print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config_path')
    parser.add_argument('--metrics', nargs='*', type=str, help='metrics to plot')
    args = parser.parse_args()

    params = {}
    main(args.config_path, metrics=args.metrics)
