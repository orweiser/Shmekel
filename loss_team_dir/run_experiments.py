from loss_team_dir.pipe_line import experiment, get_exp_name
from loss_team_dir.pipe_line.analysis import get_saved_experiments_names_paths


dataset_name = 'mnist'
resolution = 10


def _prediction_dependent_loop(without_uncertainty=False):
    # "a" is the reduced punishment for being uncertain
    a_range = [i / resolution for i in range(resolution)]

    # "b" is the incentive of being uncertain
    b_range = [i / resolution for i in range(resolution)] if not without_uncertainty else [1]

    for a in a_range:
        for b in b_range:
            yield (1, a, b)


def _reinforce_loop(without_uncertainty=False):
    # "a" is the reward for being wrong
    a_range = [-2 * i / resolution for i in range(resolution)]

    # "b" is the reward for being uncertain
    b_range = [2 * i / resolution - 1 for i in range(resolution)] if not without_uncertainty else [1]

    for a in a_range:
        for b in b_range:
            yield (1, a, b)


def _categorical_crossentropy_loop():
    yield 1


noise_range = [5 * i / resolution for i in range(resolution)]
usr = input('enter your first name please..')
noise_range = {
    'dror': noise_range[:4],
    'roee': noise_range[4:7],
    'eden': noise_range[7:10],
}[usr]
print('user noise range:', noise_range)

for noise_level in noise_range:
    for loss in [
        'categorical_crossentropy', 'prediction_dependent', 'log_reinforce', 'linear_reinforce'
    ]:
        for flag in [False, True]:
            if flag and loss == 'prediction_dependent':
                continue
            gen = {
                'categorical_crossentropy': _categorical_crossentropy_loop(),
                'prediction_dependent': _prediction_dependent_loop(without_uncertainty=flag),
                'log_reinforce': _reinforce_loop(without_uncertainty=flag),
                'linear_reinforce': _reinforce_loop(without_uncertainty=flag),
            }[loss]
            for y in gen:
                loss_params = {
                    'loss': loss, 'hyper_parameters': y, 'without_uncertainty': flag
                }

                saved_experiments, _ = get_saved_experiments_names_paths()
                exp_name = get_exp_name(dataset_name=dataset_name, noise_level=noise_level, loss_params=loss_params)

                if exp_name in saved_experiments:
                    print('Skipping experiment:', exp_name)
                    continue
                print('Starting experiment:', exp_name, '-' * 50, sep='\n')

                experiment(loss_params=loss_params, noise_level=noise_level, dataset=dataset_name,
                           save_history=True, save_weights=False,
                           batch_size=1024, epochs=30, clear=False)

                print('\n')
