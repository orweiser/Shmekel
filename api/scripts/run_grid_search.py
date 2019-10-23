import argparse
import os
import sys
sys.path.append(os.path.abspath('.'))
from api.core.grid_search import GridSearch


def main(template_path, create_configs):
    gs = GridSearch(template_path)

    if create_configs:
        gs.create_config_files()

    for exp in gs:
        exp.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('template_path')
    parser.add_argument('-dcc', '--dont_create_configs', action='store_true')

    args = parser.parse_args()

    main(args.template_path, not args.dont_create_configs)
