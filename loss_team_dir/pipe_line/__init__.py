"""
Loss team's pipeline to train, test and analyze models&losses
"""
from copy import deepcopy as copy
from .datasets import get_data_generators
from .models import fully_connected
from .losses import get_loss
from .metrics import get_metrics
from .callbacks import get_callbacks


_experiments_dir = ''  # todo: add a folder for test results that is ignored by git

_default_model_params = {
    'depth': 1,
    'width': 32,
    'base_activation': 'relu',
    'output_activation': 'softmax'
}
_default_loss_params = {
    'loss_name': 'categorical_crossentropy'
}


def get_model(**model_params):
    # todo: add functionality
    return fully_connected(**model_params)


def compile_model(model, *loss_args, **loss_kwargs):
    optimizer = 'adam'

    loss = get_loss(*loss_args, **loss_kwargs)
    metrics = get_metrics()

    model.compile(optimizer, loss=loss, metrics=metrics)


def get_exp_name(loss_name, noise_level, model_name):
    if noise_level is None:
        noise_level = 0
    noise_level = str(noise_level)

    return model_name + '_loss_' + loss_name + '_noise_level_' + noise_level


def get_model_name(**model_params):
    if 'name' in model_params.keys():
        return model_params['name']

    # todo: decide on better names
    return 'fully_connected_depth_' + str(model_params['depth']) + \
           '_width_' + str(model_params['width'])


def experiment(model_params=None, dataset='mnist',
               save_history=True, save_weights=False, save_dir=None,
               batch_size=1024, epochs=50,
               noise_level=None, **loss_params):

    loss_kwargs = copy(_default_loss_params)
    loss_kwargs.update(loss_params or {})

    model_kwargs = copy(_default_model_params)
    model_kwargs.update(model_params or {})
    model_kwargs['dataset_name'] = dataset
    model_kwargs['name'] = get_model_name(**model_kwargs)

    model = get_model(**model_kwargs)
    exp = get_exp_name(loss_kwargs['loss_name'], noise_level, model_name=model.name)
    print('Starting experiment:', exp)

    compile_model(model, **loss_kwargs)

    if isinstance(dataset, str):
        train_gen, val_gen = get_data_generators(dataset, batch_size=batch_size, noise_level=noise_level)
    elif isinstance(dataset, tuple) and isinstance(dataset, type((i for i in range(8)))):
        train_gen, val_gen = dataset
    else:
        raise Exception('Parameter "dataset" must be a string ir a tuple of two generators')

    steps_per_epoch = 60000 // batch_size
    validation_steps = 10000 // batch_size
    callbacks = get_callbacks()

    h = model.fit_generator(train_gen, steps_per_epoch, epochs, verbose=2, callbacks=callbacks,
                            validation_data=val_gen, validation_steps=validation_steps)

    if save_history or save_weights or save_dir:
        # exp_dir = save_dir or os.path.join(_experiments_dir, exp)
        # todo: handle the whole saving part
        pass

    return h
