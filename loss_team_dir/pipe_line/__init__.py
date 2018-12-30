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
    'loss_name': 'categorical_crossentropy',
    'hyper_parameters': 1,
    'minimize': True
}


def get_model(**model_params):
    """
    a function that returns a keras model according to some params.
    to be implemented
    :param model_params:
    :return:
    """
    # todo: implement
    return fully_connected(**model_params)


def compile_model(model, **loss_kwargs):
    """
    compiles a model according to specific loss.
    a wrapper around model.compile that automatically adds relevant metrics and loss tensors
    :param model: model to be compiled
    :param loss_args:
    :param loss_kwargs:
    :return:
    """
    optimizer = 'adam'

    loss = get_loss(**loss_kwargs)
    metrics = get_metrics(loss_name=loss_kwargs['loss_name'])

    model.compile(optimizer, loss=loss, metrics=metrics)


def get_exp_name(loss_params, noise_level, model_name):
    """
    name an experiment according to the hyper parameters.
    todo: decide on names and hyper-parameters to consider
    :param loss_name:
    :param noise_level:
    :param model_name:
    :return:
    """
    if noise_level is None:
        noise_level = 0
    noise_level = str(noise_level)

    loss_name, loss_weights = [loss_params[k] for k in ['loss_name', 'hyper_parameters']]

    return model_name + '_loss_' + loss_name + '_' + str(loss_weights) + '_noise_level_' + noise_level


def get_model_name(**model_params):
    """
    names a model according to it's parameters
    # todo: decide on better model names
    :param model_params:
    :return:
    """
    if 'name' in model_params.keys():
        return model_params['name']

    return 'fully_connected_depth_' + str(model_params['depth']) + \
           '_width_' + str(model_params['width'])


def adjust_params(loss_params, model_params):
    if loss_params['loss_name'] != 'categorical_crossentropy':
        model_params['output_shape'] = (11,)
    else:
        model_params['output_shape'] = (10,)


def experiment(model_params=None, loss_params=None, noise_level=None, dataset='mnist',
               save_history=True, save_weights=False, save_dir=None,
               batch_size=1024, epochs=50):
    """

    :param model_params:
    :param loss_params:
    :param dataset: 'mnist', 'cifar10' or 'fashion' for "fashion mnist"
    :param save_history: boolean.
    :param save_weights: boolean.
    :param save_dir: optional. if specified all savings will be done to this dir. otherwise, saving will be done
        to a new dir according to the experiment name
    :param batch_size: batch size for training
    :param epochs: number of epochs for training. let's have it fix on 50 for now
    :param noise_level: either None or a positive float that is the std of the noise (gaussian zero-mean)
    :return: a keras history module
    """

    loss_kwargs = copy(_default_loss_params)
    loss_kwargs.update(loss_params or {})

    model_kwargs = copy(_default_model_params)
    model_kwargs.update(model_params or {})
    model_kwargs['dataset_name'] = dataset
    model_kwargs['name'] = get_model_name(**model_kwargs)

    adjust_params(loss_kwargs, model_kwargs)

    model = get_model(**model_kwargs)
    exp = get_exp_name(loss_kwargs, noise_level, model_name=model.name)
    print('Starting experiment:', exp)

    compile_model(model, **loss_kwargs)

    if isinstance(dataset, str):
        train_gen, val_gen = get_data_generators(dataset, batch_size=batch_size, noise_level=noise_level,
                                                 num_classes=model_kwargs['output_shape'][-1])
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
