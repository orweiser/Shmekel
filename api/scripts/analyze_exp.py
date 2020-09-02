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
    args = parser.parse_args()

    params = {}
    main(args.config_paths, metrics=args.metrics)