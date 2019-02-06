"""
Loss team's pipeline to train, test and analyze models&losses
"""
from copy import deepcopy as copy
from .datasets import get_data_generators
from .models import fully_connected
from .losses import get_loss
from .metrics import get_metrics
from .callbacks import get_callbacks
import os
import pickle


def __experiments_dir():
    if os.path.exists(os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, 'loss_results_path.txt'))):
        with open(os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, 'loss_results_path.txt')), 'r') as f:
            a = [i for i in f][0]
        return a

    else:
        raise Exception('please create the file:',
                        os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, 'loss_results_path.txt')),
                        'with the results folder path in it.')


_experiments_dir = __experiments_dir()
name_sep = '--'
num_dig = 2

_default_model_params = {
    'model_type': 'fully_connected',
    'depth': 1,
    'width': 32,
    'base_activation': 'relu',
    'output_activation': 'softmax'
}
_default_loss_params = {
    'loss': 'categorical_crossentropy',
    'hyper_parameters': 1,
    'minimize': True,
    'without_uncertainty': False
}


def __str_2_num(s):
    x = float(s)
    if x == round(x):
        x = round(x)
    return x


def get_model(model_type=None, **model_params):
    """
    a function that returns a keras model according to some params.
    to be implemented
    :param model_params:
    :return:
    """
    if model_type is None or model_type == 'fully_connected':
        return fully_connected(**model_params)

    else:
        raise Exception('only fully_connected model type os supported')


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
    metrics = get_metrics(without_uncertainty=loss_kwargs['without_uncertainty'])

    model.compile(optimizer, loss=loss, metrics=metrics)


def get_exp_name(dataset_name, noise_level, loss_params=_default_loss_params, model_params=_default_model_params, name_sep=name_sep):
    """
    name an experiment according to the hyper parameters.
    :param loss_name:
    :param noise_level:
    :param model_name:
    :return:
    """
    if noise_level is None:
        noise_level = 0

    model_name = get_model_name(**model_params)
    loss_name = get_loss_name(**loss_params)

    name = dataset_name
    name += name_sep + 'model_' + model_name
    name += name_sep + 'loss_' + loss_name
    name += name_sep + 'noise_level_' + str(round(noise_level, num_dig))

    return name


def _exp_name_to_params(exp_name, name_sep=name_sep):
    dataset, model, loss, noise = exp_name.split(name_sep)
    model_params = _model_name_to_params(model.split('model_')[-1])
    loss_params = _loss_name_to_params(loss.split('loss_')[-1])
    noise_level = __str_2_num(noise.split('noise_level_')[-1])

    return model_params, loss_params, noise_level


def get_model_name(**model_params):
    """
    names a model according to it's parameters
    :param model_params:
    :return:
    """
    if 'name' in model_params.keys():
        return model_params['name']

    return model_params['model_type'] + '_depth_' + str(model_params['depth']) + \
           '_width_' + str(model_params['width'])


def _model_name_to_params(model_name):
    model_type = model_name.split('_depth')[0]
    depth = int(model_name.split('_depth_')[1].split('_width_')[0])
    width = int(model_name.split('_width_')[1])

    return {
        'model_type': model_type, 'depth': depth, 'width': width
    }


def get_loss_name(**loss_params):
    if 'name'in loss_params.keys():
        return loss_params['name']

    if type(loss_params['hyper_parameters']) is tuple:
        rounded_parameters = tuple([round(x, num_dig) for x in loss_params['hyper_parameters']])
    else:
        rounded_parameters = round(loss_params['hyper_parameters'], num_dig)

    name = loss_params['loss'] + '_weights_' + str(rounded_parameters)
    if loss_params['without_uncertainty']:
        name += '_without_uncertainty'

    return name


def _loss_name_to_params(loss_name):
    without_uncertainty = loss_name.endswith('_without_uncertainty')
    loss = loss_name.split('_weights_')[0]
    hyper_parameters = loss_name.split('_weights_')[1].split('_without_uncertainty')[0]

    if hyper_parameters[0] == '(':
        hyper_parameters = hyper_parameters[1:-1]
        hyper_parameters = hyper_parameters.replace(' ', '')
        hyper_parameters = tuple([__str_2_num(x) for x in hyper_parameters.split(',')])



    return {
        'loss': loss,
        'hyper_parameters': hyper_parameters,
        'without_uncertainty': without_uncertainty
    }


def adjust_params(loss_params, model_params):
    if loss_params['without_uncertainty']:
        model_params['output_shape'] = (10,)
    else:
        model_params['output_shape'] = (11,)


def experiment(model_params=None, loss_params=None, noise_level=None, dataset='mnist',
               save_history=True, save_weights=False, save_dir=None,
               batch_size=1024, epochs=50, clear=False, return_model=False):
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

    if return_model and clear:
        raise Exception("Can't clear session when return_model is True")

    loss_kwargs = copy(_default_loss_params)
    loss_kwargs.update(loss_params or {})

    model_kwargs = copy(_default_model_params)
    model_kwargs.update(model_params or {})
    model_kwargs['dataset_name'] = dataset
    model_kwargs['name'] = get_model_name(**model_kwargs)

    adjust_params(loss_kwargs, model_kwargs)

    model = get_model(**model_kwargs)
    exp = get_exp_name(dataset_name=dataset, noise_level=noise_level,
                       loss_params=loss_kwargs, model_params=model_kwargs)

    print('model_shapes: input', model.input_shape, ', output', model.output_shape)

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
        exp_dir = save_dir or os.path.join(_experiments_dir, exp)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        if save_weights:
            weights_path = os.path.join(exp_dir, 'weights.h5')
            model.save_weights(weights_path)
            print('weights saved at:', weights_path)

        if save_history:
            history_path = os.path.join(exp_dir, 'history.h5')

            with open(history_path, 'wb') as f:
                pickle.dump(h.history, f)
            print('history saved at:', history_path)

    if clear:
        from keras.backend import clear_session
        clear_session()

    if return_model:
        return h, model

    return h
