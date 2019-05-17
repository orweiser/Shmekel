from api import load_config, get_exp_from_config
import sys
import os


def config_path(exp_batch, exp_num):
    dir_path = os.path.join('example_project', 'configs', 'e%02d' % exp_batch)

    prefix = 'e%02d_%02d' % (exp_batch, exp_num)
    match = None
    for fname in os.listdir(dir_path):
        if fname.startswith(prefix):
            if match:
                raise ValueError('More than one config file matches the prefix %s' % prefix)
            else:
                match = os.path.join(dir_path, fname)

    return match


exp_batch = int(sys.argv[1])
exp_num = int(sys.argv[2])

config = load_config(config_path(exp_batch, exp_num))
exp = get_exp_from_config(config)

exp.run()


