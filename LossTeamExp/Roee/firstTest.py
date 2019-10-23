from api.core import Experiment
from api.core import load_config
from api.core.grid_search import GridSearch


def single_test():
    config = load_config("LossTeamExp/Roee/config.json")
    my_exp = Experiment(**config)

    my_exp.run()

    results = my_exp.results
    results.plot()


def grid_search_test():
    template_path = r"C:\shmekels\Shmekel\LossTeamExp\Roee\grisSearchConfig.json"
    configs_dir = r"C:\shmekels\Shmekel\LossTeamExp\Roee\jsons"
    grid_serach_exps = GridSearch(template_path, configs_dir)
    grid_serach_exps.create_config_files()
    for exp in grid_serach_exps:
        exp.run()
        # exp_result = exp.results
        # exp_result.plot()


if __name__ == '__main__':
    grid_search_test()
