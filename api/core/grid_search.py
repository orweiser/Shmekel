import json
import os
from copy import deepcopy as copy
import os
from api.core import get_exp_from_config, load_config


class GridSearch:
    """
    an object to handle a bunch of experiments created via grid search.
    in a nutshell, it lets you iterate over experiments in your grid search
    Examples:
        1. to run all experiments:
                for exp in grid_search:
                    exp.run()
        2. to check status:
                for exp in grid_search:
                    exp.run()
        3. to get the best epoch out of every experiment:
                best_epochs = []
                for exp in grid_search:
                    best_epochs.append(exp.results.get_best_epoch()
        etc.

    to create a grid_search you need a template. an example can be found in:
        api/example_grid_search_template.json
    you also need to state a "configs_dir" parameter, in which config files are stored
    as a first step, before you can iterate, you need to create the config files

    example:
        gs = GridSearch(template_path, configs_dir)
        gs.create_config_files()

    """
    def __init__(self, template_path, configs_dir=None):
        self.template_path = template_path
        self.configs_dir = configs_dir or os.path.split(template_path)[0]

    def read_template(self, path=None):
        path = path or self.template_path
        with open(path) as f:
            text = f.read()
        template, overrides = text.split('__OVERRIDES__')
        overrides = self.parse_overrides(overrides)
        return template, overrides

    @staticmethod
    def parse_line(line):
        if 'lambda' in line:
            var, values = line.split('lambda')
            var = var.split(':')[0].strip(' ')

            format, keys = values.split('@')
            format = format.strip(' ').replace('%', '%s')

            keys = keys.split(',')
            keys = [k.strip(' ') for k in keys]

            x = ', '.join(['"%s"' % k for k in keys])
            x = '[%s]' % x

            keys = json.loads(x)
            keys = [str(v) for v in keys]

            values = [lambda x: format % tuple([x[key] for key in keys])]

        else:
            var, values = line.split(':')
            var = var.strip(' ')

            values = json.loads('[%s]' % values)
            values = [str(v) for v in values]

        return var, values

    def parse_overrides(self, overrides):
        overrides = [l.strip() for l in overrides.splitlines() if l]
        overrides = [l for l in overrides if not l.startswith('//')]
        return [self.parse_line(l) for l in overrides if l]

    @staticmethod
    def overrides_to_combs(overrides):
        indices = [[0] * len(overrides)]
        for i, (_, options) in enumerate(overrides):
            tmp = indices
            indices = []
            for j, v in enumerate(options):
                new = [copy(t) for t in tmp]
                for n in new:
                    n[i] = j
                indices.extend(new)

        combs = [{key: options[i] for i, (key, options) in zip(ind, overrides)} for ind in indices]
        return combs

    def create_config_files(self, template_path=None, configs_dir=None):
        template_path = template_path or self.template_path
        configs_dir = configs_dir or self.configs_dir

        template, overrides = self.read_template(template_path)
        combs = self.overrides_to_combs(overrides)

        if not os.path.exists(configs_dir):
            os.makedirs(configs_dir)

        function = type(lambda: None)
        for comb in combs:
            for key, val in comb.items():
                if isinstance(val, function):
                    comb[key] = val(comb)

        paths = []
        for comb in combs:
            s = template
            for key, val in comb.items():
                s = s.replace(key, val)

            for key, val in [('False', 'false'), ('True', 'true'), ('None', 'null')]:
                s = s.replace(key, val)

            path = os.path.join(configs_dir, comb['__Name__'] + '.json')
            with open(path, 'w') as f:
                f.write(s)
            paths.append(path)

        return paths

    def iter_exps(self, configs_dir=None):
        configs_dir = configs_dir or self.configs_dir

        for f in os.listdir(configs_dir):
            if not f.endswith('.json'):
                continue

            if f.startswith('template'):
                continue

            path = os.path.join(configs_dir, f)

            config = load_config(path)

            exp = get_exp_from_config(config)

            yield exp

    def __iter__(self):
        return self.iter_exps()

