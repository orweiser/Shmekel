import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join('..', '..', '..', __file__))
from api.core import get_exp_from_config, load_config # this must be under 'sys.path.insert'


def _get_attribute_from_config(config, attr):
    attr_keys = attr.split("-")
    for key in attr_keys:
        if key in config:
            config = config[key]
        else:
            raise KeyError("attribute " + attr + " doesn't exist in config")

    return config


def _print_exp_metric(exp, metric):
    print('\n\n')
    print(exp.name, metric)
    exp.results.get_best_epoch(metric).print()


def main(paths, metrics=None, attr=None):
    exps = [get_exp_from_config(load_config(path)) for path in paths]

    metrics = metrics or ['val_acc']

    legend = []
    plt.figure()
    if attr:
        for metric in metrics:
            x = []
            y = []
            for exp in exps:
                x.append(_get_attribute_from_config(exp.config, attr))
                y.append(exp.results.get_best_epoch(metric).scores[metric])
            sorted_x = sorted(x)
            sorted_y = []
            for x_val in sorted_x:
                index = x.index(x_val)
                sorted_y.append(y[index])
                del x[index]
                del y[index]

            plt.plot(sorted_x, sorted_y)
            legend.append(f'{attr}--{metric}')
        plt.xlabel(attr)
    else:
        for exp in exps:
            for metric in metrics:
                plt.plot(range(1, len(exp.results[metric]) + 1), exp.results[metric])
                legend.append(f'{os.path.basename(exp.name).rsplit(".", 1)[0]}--{metric}')
        plt.xlabel('#Epoch')
    plt.legend(legend)
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
    parser.add_argument('--attr', type=str, help='plot best epoch metrics by attribute')
    args = parser.parse_args()

    params = {}
    main(args.config_paths, metrics=args.metrics, attr=args.attr)
