import json
import os
from copy import deepcopy as copy
import os
from api.core import get_exp_from_config, load_config


function = type(lambda: None)


class GridSearch:
    def __init__(self, template_path, configs_dir):
        self.template_path = template_path
        self.configs_dir = configs_dir

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
            # WARNING ...
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

            print(keys, format)

            values = [lambda x: format % tuple([x[key] for key in keys])]

        else:
            var, values = line.split(':')
            var = var.strip(' ')

            values = json.loads('[%s]' % values)
            values = [str(v) for v in values]

        return var, values

    def parse_overrides(self, overrides):
        return [self.parse_line(l) for l in overrides.splitlines() if l]

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

            path = os.path.join(configs_dir, f)

            config = load_config(path)

            exp = get_exp_from_config(config)

            yield exp

    def __iter__(self):
        return self.iter_exps()

