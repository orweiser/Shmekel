import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join('..', '..', '..', __file__))
from api.core import get_exp_from_config, load_config # this must be under 'sys.path.insert'


def _get_attribute_from_config(config, attr, sep):
    attr_keys = attr.split(sep)
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


def main(paths, metrics=None, attr=None, sep='-'):
    exps = [get_exp_from_config(load_config(path)) for path in paths]

    metrics = metrics or ['loss']

    legend = []
    plt.figure()
    if attr:
        for metric in metrics:
            x = []
            y = []
            for exp in exps:
                x.append(_get_attribute_from_config(exp.config, attr, sep=sep))
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


# specific for depth/width variation, and val_acc/runtime metrics
def scatter_map(paths):
    exps = [get_exp_from_config(load_config(path)) for path in paths]

    metrics = ['acc', 'runtime']

    fig, ax = plt.subplots()
    x = []
    y = []
    sizes = []
    colors = []
    for exp in exps:
        if (exp.results):
            best_epoch = exp.results.get_best_epoch(metrics[0])
        else:
            continue
        x.append(exp.get_avg_runtime())
        y.append(best_epoch['val_acc'])
        sizes.append(exp.model_config['width'])
        colors.append(exp.model_config['depth'])

    scatter = plt.scatter(x, y, s=sizes, c=colors, alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(prop="sizes"), loc="lower left", title="width")
    ax.add_artist(legend1)
    legend2 = ax.legend(*scatter.legend_elements(prop="colors"), loc="upper right", title="depth")
    ax.add_artist(legend2)
    plt.xlabel('runtime')
    plt.ylabel('val_acc')
    plt.title('acc vs runtime comparison')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config_paths', nargs='+')
    parser.add_argument('--metrics', nargs='*', type=str, help='metrics to plot')
    parser.add_argument('--attr', type=str, help='plot best epoch metrics by attribute')
    parser.add_argument('--sep', type=str, default='-', help='separator of keys. default is  "-"')
    args = parser.parse_args()

    params = {}
    main(args.config_paths, metrics=args.metrics, attr=args.attr, sep=args.sep)
