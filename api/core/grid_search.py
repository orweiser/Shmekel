import json
import os
from copy import deepcopy as copy
import os
from api.core import get_exp_from_config, load_config
from Utils.logger import logger
import matplotlib.pyplot as plt
import numpy as np


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

        function = type(lambda: None)
        for comb in combs:
            for key, val in comb.items():
                if isinstance(val, function):
                    comb[key] = val(comb)

        if not os.path.exists(configs_dir):
            os.makedirs(configs_dir)

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
    def __init__(self, configs_dir):
        self.configs_dir = configs_dir

    def iter_exps(self, configs_dir=None):
        configs_dir = configs_dir or self.configs_dir

        for f in os.listdir(configs_dir):
            if not f.endswith('.json'):
                continue

            path = os.path.join(configs_dir, f)

            config = load_config(path)

            identifiers = config.pop('identifiers')

            exp = get_exp_from_config(config)
            exp.identifiers = identifiers
            yield exp

    def __iter__(self):
        return self.iter_exps()

    def __iter_fixed_contained(self, mode, fixed_values: dict):
        assert mode in ['fixed', 'contained']

        def is_to_skip(val1, val2):
            if mode == 'fixed':
                return val1 != val2
            else:
                return val1 not in val2

        for exp in self:
            to_continue = False
            for key, val in fixed_values.items():
                if is_to_skip(exp.identifiers[key], val):
                    to_continue = True
                    break

            if to_continue:
                continue

            else:
                yield exp

    def iter_fixed(self, fixed_values: dict):
        for x in self.__iter_fixed_contained(mode='fixed', fixed_values=fixed_values):
            yield x

    def iter_contained(self, fixed_values: dict):
        for x in self.__iter_fixed_contained(mode='contained', fixed_values=fixed_values):
            yield x

    def iter_modulo(self, mod=4, rem=0):
        for i, x in enumerate(self.iter_exps()):
            if i % mod == rem:
                yield x

    def plot_a_slice(self, fixed_values: dict, metric='val_acc', title=None):
        fig = plt.figure()
        legend = []
        for exp in self.iter_fixed(fixed_values=fixed_values):
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

        minimal = {}
        for e in self:      # maybe obsolete when using from plot all parameter slices
            for key, val in e.identifiers.items():
                minimal.setdefault(key, set()).add(val)

        slices = slices or minimal[param]

        new_slices = []
        for s in slices:
            try:
                s = str(s)
                new_slices.append(json.loads(s))
            except json.decoder.JSONDecodeError:
                new_slices.append(s)

        for val in new_slices:
            title = 'Slice - %s (%s)' % (param, str(val))
            fig = self.plot_a_slice({param: val}, metric=metric, title=title)

            if figs_dir:
                fig.savefig(os.path.join(figs_dir, title + '.png'))

    def plot_all_parameters_slices(self, metric='val_acc', figs_dir=None):
        minimal = {}
        for e in self:
            for key, val in e.identifiers.items():
                minimal.setdefault(key, set()).add(val)

        for param in minimal.keys():
            self.plot_parameter_slices(param, metric=metric, figs_dir=figs_dir)

    def print_slice_information(self, compare, slices=None, fixed_values={}, metric='val_acc', file=None):

        if slices is None:
            slices = {}
            for e in self.iter_fixed(fixed_values):
                for key, val in e.identifiers.items():
                    slices.setdefault(key, set()).add(val)

        slices = slices[compare]
        new_slices = []
        for s in slices:
            try:
                s = str(s)
                new_slices.append(json.loads(s))
            except json.decoder.JSONDecodeError:
                new_slices.append(s)

        for val in new_slices:
            curr_slice = []
            fixed_values[compare] = val
            for exp in self.iter_fixed(fixed_values):
                if exp.history:
                    curr_slice.append(max(exp.results[metric]))
            best = max(curr_slice)
            size = len(curr_slice)
            avg = np.mean(curr_slice)
            print('Slice - %s (%s):' % (compare, str(val)))
            print('total nets: %d;  best: %d;   avg: %d' % (size, best, avg))
