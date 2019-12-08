import os
import warnings


def get_config(config_path):
    def _read_a_file(fname):
        with open(fname) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        return [x.strip() for x in content]

    user_config = {
        'data_path': None,
        'shmekel_drive_path': None
    }

    __None_str = ''
    __item_separator = ','

    if not os.path.isfile(config_path):
        raise Exception("There is no file in the location given %s" % config_path)

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
            warning = '\nconfiguration field: "' + key + '" is empty.\nPlease fill in details in config file at:\n' + \
                       config_path
            warnings.warn(warning, Warning)

    return config


def get_data_path():
    config_path = 'Shmekel_config.txt'
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.pardir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError('missing file "Shmekel_config.txt"')

    config = get_config(config_path)
    data_path = config["data_path"]
    return os.path.join(data_path, 'Stocks')


PATTERN = ('Open', 'High', 'Low', 'Close', 'Volume')
FEATURE_AXIS = -1
DATA_PATH = get_data_path()
