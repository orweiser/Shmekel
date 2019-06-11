import os
from api.core import get_exp_from_config, load_config


ConfigsDir = r'C:\Users\Danielle\PycharmProjects\Shmekel\projects\dror\preprocess\e01_dense_model\configs'


for f in os.listdir(ConfigsDir):
    if not f.endswith('.json'):
        continue

    path = os.path.join(ConfigsDir, f)
    config = load_config(path)

    exp = get_exp_from_config(config)

    exp.run()
