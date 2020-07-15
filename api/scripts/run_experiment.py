import argparse
import os
import sys
sys.path.insert(0, os.path.join('..', '..', '..', __file__))
from api.core import get_exp_from_config, load_config


def main(path):
    config = load_config(path)
    exp = get_exp_from_config(config)

    exp.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config_path')
    args = parser.parse_args()

    main(args.config_path)
