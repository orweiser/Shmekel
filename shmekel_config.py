import os
import time
import warnings


default_config = {
    'stock_types': ('ETFs', 'Stocks'),
    'pattern': ('Open', 'High', 'Low', 'Close', 'Volume'),
    'feature_axis': 0,
    'stock_data_file_ending': '_list.pickle'
}
user_paths_config = {
    'data_path': None,
    'shmekel_drive_path': None
}


__None_str = ''
__item_separator = ','
__config_file_name = 'Shmekel_config.txt'
__config_file_location = os.path.abspath(os.path.pardir)

config_path = os.path.join(__config_file_location, __config_file_name)


def get_line(key, value):
    line = key + '='
    if isinstance(value, (list, tuple)):
        for x in value:
            line += x + __item_separator
    elif value is None:
        line += __None_str
    else:
        line += str(value)
    line += '\n'

    return line


def write_config(config=None):
    config = config or {**default_config, **user_paths_config}
    with open(config_path, 'w') as f:
        for key, item in config.items():
            f.write(get_line(key, item))


def _read_a_file(fname):
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    return [x.strip() for x in content]


def get_config():
    config = {}
    for line in _read_a_file(config_path):
        if not len(line):
            continue

        s = line.split('=')
        if len(s) == 1:
            config[s[0]] = None
        elif len(s) == 2:
            rest = s[1]
            if not rest:
                config[s[0]] = None
                continue

            if len(rest.split(__item_separator)) > 1:
                config[s[0]] = tuple([x for x in rest.split(__item_separator) if x])
            else:
                config[s[0]] = rest.replace(__item_separator, '')

        if config[s[0]] == __None_str:
            config[s[0]] = None

    for key, item in config.items():
        if not item:
            warning = '\nconfiguration field: "' + key + '" is empty.\nPlease fill in details in config file at:\n' + config_path
            warnings.warn(warning, Warning)

    config['feature_axis'] = int(config['feature_axis'][0])

    time.sleep(0.1)
    return config


def add_param_to_config(name, value):
    with open(config_path, 'a') as f:
        f.write(get_line(name, value))


if not os.path.exists(config_path):
    write_config()

    warning = '\nno configuration file was found at\n' + config_path + '.\n' + \
              'created new config file - to be filled by user'
    warnings.warn(
        warning, Warning, 0
    )


