from api.core import Experiment
from api.core import load_config

config = load_config("config.json")
my_exp = Experiment(**config)

my_exp.run()


