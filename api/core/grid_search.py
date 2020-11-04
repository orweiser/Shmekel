import json
import os
from copy import deepcopy as copy
import os
from api.core import get_exp_from_config, load_config
from Utils.logger import logger
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from copy import deepcopy


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
                    exp.status
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

            values = [lambda x: (format % tuple([x[key] for key in keys])).replace('"', '')]

        else:
            var, *values = line.split(':')
            values = ':'.join(values)
            var = var.strip(' ')

            values = json.loads('[%s]' % values)
            values = [json.dumps(v) for v in values]

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

    def generate_configs(self):
        template_path = self.template_path

        template, overrides = self.read_template(template_path)
        combs = self.overrides_to_combs(overrides)

        function = type(lambda: None)
        for comb in combs:
            for key, val in comb.items():
                if isinstance(val, function):
                    comb[key] = val(comb)

        for comb in combs:
            s = self.get_final_config_string(comb, template)
            yield json.loads(s)

    def create_config_files(self, template_path=None, configs_dir=None):
        template_path = template_path or self.template_path
        configs_dir = configs_dir or self.configs_dir

        template, overrides = self.read_template(template_path)
        combs = self.overrides_to_combs(overrides)

        function = type(lambda: None)
        for comb in combs:
            for key, val in comb.items():
                if isinstance(val, function):
                    comb[key] = val(comb)

        if not os.path.exists(configs_dir):
            os.makedirs(configs_dir)

        paths = []
        for comb in combs:
            s = self.get_final_config_string(comb, template)

            path = os.path.join(configs_dir, comb['__Name__'] + '.json').replace('"', '')
            with open(path, 'w') as f:
                f.write(s)
            paths.append(path)

        return paths

    @staticmethod
    def get_final_config_string(comb, template):
        s = template
        for key, val in comb.items():
            s = s.replace(key, val)
        for key, val in [('False', 'false'), ('True', 'true'), ('None', 'null')]:
            s = s.replace(key, val)
        return s

    def get_all_combs(self, parse_lambdas=True):
        template_path = self.template_path

        template, overrides = self.read_template(template_path)
        combs = self.overrides_to_combs(overrides)

        if parse_lambdas:
            function = type(lambda: None)
            for comb in combs:
                for key, val in comb.items():
                    if isinstance(val, function):
                        comb[key] = val(comb)

        return combs

    @staticmethod
    def combs_2_minimal_keys_values(combs):
        all_options = {}

        for comb in combs:
            for key, val in comb.items():
                all_options.setdefault(key, set()).add(val)

        minimal = {key: val for key, val in all_options.items() if len(val) > 1}
        return minimal

    def get_exp_name_2_combs_converter(self, minimal=True):
        combs = self.get_all_combs()

        if minimal:
            non_degenerate_keys = [k for k in self.combs_2_minimal_keys_values(combs).keys()]
            combs = [{key: c[key] for key in non_degenerate_keys} for c in combs]

        converter = {}
        for comb in combs:
            try:
                name = comb['__Name__']
            except KeyError:
                raise AssertionError('this method assumes that all combinations include a "__Name__" field')

            if name in converter:
                raise RuntimeError('Collision! two experiments with the same name [%s]' % name)

            parsed_comb = {}
            for key, val in comb.items():
                try:
                    parsed_comb[key] = json.loads(val)
                except json.decoder.JSONDecodeError:
                    parsed_comb[key] = val

            converter[name] = parsed_comb

        return converter

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

    def __iter_fixed_contained(self, mode, fixed_values: dict, minimal=True, yield_comb=False):
        assert mode in ['fixed', 'contained']

        def is_to_skip(val1, val2):
            if mode == 'fixed':
                return val1 != val2
            else:
                return val1 not in val2

        converter = self.get_exp_name_2_combs_converter(minimal=minimal)
        for exp in self:
            comb = converter[exp.name]

            to_continue = False
            for key, val in fixed_values.items():
                if is_to_skip(comb[key], val):
                    to_continue = True
                    break

            if to_continue:
                continue

            if yield_comb:
                yield exp, comb
            else:
                yield exp

    def iter_fixed(self, fixed_values: dict, minimal=True, yield_comb=False):
        for x in self.__iter_fixed_contained(mode='fixed', fixed_values=fixed_values,
                                             minimal=minimal, yield_comb=yield_comb):
            yield x

    def iter_contained(self, fixed_values: dict, minimal=True, yield_comb=False):
        for x in self.__iter_fixed_contained(mode='contained', fixed_values=fixed_values,
                                             minimal=minimal, yield_comb=yield_comb):
            yield x

    def plot_a_slice(self, fixed_values: dict, metric='val_acc', title=None, minimal=True):
        fig = plt.figure()
        legend = []
        for exp in self.iter_fixed(fixed_values=fixed_values, minimal=minimal):
            if not exp.results:
                logger.error('trying to plot an experiment with no result[%s]. Skipping', exp.name)
                continue

            legend.append(exp.name)
            plt.plot(exp.results[metric])

        plt.grid('on')
        plt.xlabel('# Epochs')
        plt.ylabel(metric)
        if title:
            plt.title(title)
        plt.legend(legend)

        return fig

    def plot_parameter_slices(self, param, slices=None, metric='val_acc', figs_dir=None):
        if figs_dir:
            if not os.path.exists(figs_dir):
                os.makedirs(figs_dir)

        combs = self.get_all_combs(parse_lambdas=False)
        minimal = self.combs_2_minimal_keys_values(combs)

        slices = slices or minimal[param]

        new_slices = []
        for s in slices:
            try:
                new_slices.append(json.loads(s))
            except json.decoder.JSONDecodeError:
                new_slices.append(s)

        for val in new_slices:
            title = 'Slice - %s (%s)' % (param, str(val))
            fig = self.plot_a_slice({param: val}, metric=metric, title=title)

            if figs_dir:
                fig.savefig(os.path.join(figs_dir, title + '.png'))

    def plot_all_parameters_slices(self, metric='val_acc', figs_dir=None):
        combs = self.get_all_combs(parse_lambdas=False)
        minimal = self.combs_2_minimal_keys_values(combs)

        for param in minimal.keys():
            self.plot_parameter_slices(param, metric=metric, figs_dir=figs_dir)


class GridSearch2:
    def __init__(self, template_path):
        self.template_path = template_path
        self._configs_iterator = JsonGrid(template_path, infer_name=True)

        if len(self._configs_iterator) > 10000:
            logger.warning('Large Grid Search. Working in memory efficient mode')
        else:
            self._configs_iterator = list(self._configs_iterator)

    def __iter__(self):
        for config in self._configs_iterator:
            yield get_exp_from_config(config)

    def __len__(self):
        return len(self._configs_iterator)

    def run(self):
        for exp in self:
            exp.run()


class JsonGrid:
    """
    this class takes a json template file as input
    and creates a grid of parsed json (native python)
    according to the template configuration

    Note:
        input file is assumed to represent a dictionary

    Variables:
        This framework allows you to define variables
        in your json file. variables are defined with
        all of their possible values for the grid.

        all variables are a key-value pair such that:
            1. key starts and ends with "__"
            2. value is actually a list of possible values
            3. all variables are defined in the top level of the json file

        for example, if:
            "__X__": [1, 2, 3]
            "__Y__": [4, 5]
        then we'll get a 3 by 2 grid with all the combinations of
        values for __X__ and __Y__

        Variables can be used anywhere in the json file

    Example:
        {
            "combination": ["__X__", "__Y__"],
            "__X__": [1, 2, 3],
            "__Y__": [4, 5]
        }

        will yield the following 6 dictionaries:
            {"combination": [1, 4]}
            {"combination": [1, 5]}
            {"combination": [2, 4]}
            {"combination": [2, 5]}
            {"combination": [3, 4]}
            {"combination": [3, 5]}

    """
    def __init__(self, template_path, infer_name=True):
        self._template_path = template_path

        self.content = None
        self.variables = None
        self.variable_to_appearances_mapping = None
        self.infer_name = infer_name

        self._build()

    def __len__(self):
        lengths = [len(var) for var in self.variables.values()]
        return int(np.prod(lengths))

    def generate_combinations(self):
        def iter_options(_options):
            assert isinstance(_options, list)
            yield from _options

        variable_names, variable_options_list = zip(*map(tuple, self.variables.items()))
        iterators = map(iter_options, variable_options_list)

        for values in product(*iterators):
            yield {name: val for name, val in zip(variable_names, values)}

    def comb_to_config(self, comb):
        config = deepcopy(self.content)

        # insert standalone variables
        for variable, value in comb.items():
            for keys in self.variable_to_appearances_mapping.get(variable, []):
                this_config = config

                for key in keys[:-1]:
                    try:
                        this_config = this_config[key]
                    except:
                        print('')
                key = keys[-1]
                this_config[key] = value

        # insert in-string variables
        config_string = json.dumps(config)
        for variable, value in comb.items():
            config_string = config_string.replace(variable, str(value))
        config = json.loads(config_string)

        return config

    def generate_configs(self):
        yield from map(self.comb_to_config, self.generate_combinations())

    __iter__ = generate_configs

    def _load(self):
        with open(self._template_path) as f:
            full_template = json.load(f)

        variables = {key: val for key, val in full_template.items() if self._is_variable(key)}
        content = {key: val for key, val in full_template.items() if not self._is_variable(key)}

        return content, variables

    def _build(self):
        self.content, self.variables = self._load()

        if self.infer_name and 'name' not in self.content:
            suffix = self.generate_name_suffix(self.variables)
            path = self.get_relative_template_path()
            self.content['name'] = os.path.join(path, suffix)

        self.variable_to_appearances_mapping = self.find_variables_appearances(self.content, self.variables)

    def get_relative_template_path(self):
        path = self._template_path

        head = True
        path_parts = []
        while head:
            head, tail = os.path.split(path)
            path_parts.append(tail)
            path = head
        path_parts = list(reversed(path_parts))

        assert 'configs' in path_parts, '"template_path" must be inside the "configs" dir'
        path_parts = path_parts[(path_parts.index('configs') + 1):]

        return os.path.join(*path_parts)

    @staticmethod
    def generate_name_suffix(variables, big_sep='--', small_sep='_'):
        def get_initials(_stripped_var):
            capitals = [c for c in _stripped_var if c == c.upper()]
            return ''.join(capitals)

        def var2name(_var):
            initials = get_initials(_stripped_var=_var.strip('__'))
            return small_sep.join((initials, _var))

        return big_sep.join(var2name(var) for var in variables)

    @staticmethod
    def _is_variable(name):
        return name.startswith('__') and name.endswith('__')

    @classmethod
    def _generate_variable_keys(cls, item):
        if isinstance(item, dict):
            iterator = item.items()
        elif isinstance(item, list):
            iterator = enumerate(item)
        else:
            raise RuntimeError()

        for key, val in iterator:
            if isinstance(val, str) and cls._is_variable(val):
                yield (key, val)

            elif isinstance(val, (dict, list)):
                for keys in cls._generate_variable_keys(val):
                    yield (key,) + keys

    @classmethod
    def find_variables_appearances(cls, content, variables):
        mapping = {}
        for keys in cls._generate_variable_keys(content):
            if keys[-1] in variables:
                mapping.setdefault(keys[-1], []).append(keys[:-1])
        return mapping
