import os
from itertools import product


def get_all_combinations(dict):
    keys = [k for k in dict.keys()]
    for values in product(*[dict[key] for key in keys]):
        yield {key: val for key, val in zip(keys, values)}


TEMPLATE_PATH = os.path.join('example_project', 'configs', 'e01', 'TEMPLATE.json')
OUTPUT_DIR = os.path.split(TEMPLATE_PATH)[0]

CONFIG_FILENAME_TEMPLATE = 'e01_%02d--%s.json'


"""
    strings that should be replaced in the new config files:
        - ui:
            EXP_NUMBER
            EXP_DESCRIPTION
        
        - grid parameters:
            NUM_MODEL_UNITS
            TIME_SAMPLE_LENGTH
            OPTIMIZER
"""


grid_parameters = {
    'NUM_MODEL_UNITS': [8, 16],
    'TIME_SAMPLE_LENGTH': [1, 2],
    'OPTIMIZER': ['"adam"', '"sgd"']
}

with open(TEMPLATE_PATH) as f:
    template_config = f.read()

for cnt, comb in enumerate(get_all_combinations(grid_parameters)):
    exp_num = cnt + 1
    num_model_units, time_sample_length, optimizer = [comb[key] for key in [
        'NUM_MODEL_UNITS', 'TIME_SAMPLE_LENGTH', 'OPTIMIZER'
    ]]

    exp_description = ('units_%d-time_%d-optimizer_%s' % tuple([
        comb[key] for key in ['NUM_MODEL_UNITS', 'TIME_SAMPLE_LENGTH', 'OPTIMIZER']
    ])).replace('"', '')

    new_config_path = os.path.join(OUTPUT_DIR, CONFIG_FILENAME_TEMPLATE % (exp_num, exp_description))

    config = template_config.replace('EXP_NUMBER', str(exp_num))
    config = config.replace('EXP_DESCRIPTION', exp_description)
    for key in grid_parameters:
        config = config.replace(key, str(comb[key]))

    print(new_config_path)
    with open(new_config_path, 'w') as f:
        f.write(config)
