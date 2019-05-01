from api.core import Experiment
from api.core import load_config

config = load_config(r"C:\shmekels\Shmekel\Test\Roee\config.json")
my_exp = Experiment(**config)

my_exp.run()


