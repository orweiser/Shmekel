import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join('..', '..', '..', __file__))
from api.core import get_exp_from_config, load_config


def _print_exp_metric(exp, metric):
    print('\n\n')
    print(exp.name, metric)
    exp.results.get_best_epoch(metric).print()


def get_legend_name(name):
    return name[name.rfind('/')+1: name.rfind('.')]


def main(paths, metrics=None):
    exps = [get_exp_from_config(load_config(path)) for path in paths]

    metrics = metrics or ['val_acc']

    legend = []
    plt.figure()
    for exp in exps:
        for metric in metrics:
            plt.plot(range(1, len(exp.results[metric]) + 1), exp.results[metric])
            legend.append(f'{exp}--{metric}')
    plt.legend(legend)
    plt.xlabel('#Epoch')
    plt.title('Learning Curves Comparison')
    plt.grid()
    plt.show()

    for exp in exps:
        for metric in metrics:
            _print_exp_metric(exp, metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config_paths', nargs='+')
    parser.add_argument('--metrics', nargs='*', type=str, help='metrics to plot')
    args = parser.parse_args()

    params = {}
    main(args.config_paths, metrics=args.metrics)
