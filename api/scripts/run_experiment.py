import argparse
import os
import sys
sys.path.insert(0, os.path.join('..', '..', '..', __file__))
from api.core import get_exp_from_config, load_config


def main(paths, no_run, print_status):
    for path in paths:
        config = load_config(path)
        exp = get_exp_from_config(config)

        if print_status:
            print(f'Exp: {path}\nStatus: {exp.status}')

        if not no_run:
            try:
                exp.run()
            except:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config_paths', nargs='+')
    parser.add_argument('-nr', '--no_run', action='store_true')
    parser.add_argument('-ps', '--print_status', action='store_true')
    args = parser.parse_args()

    main(args.config_paths, no_run=args.no_run, print_status=args.print_status)
