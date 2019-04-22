import json
import numpy as np
from api import core

MIN_SIZE = 3
MAX_SIZE = 7
MAX_NUM_OF_LAYERS = 10
LAYERS_TYPES = ['KerasLSTM', 'Dense']
ALL_DENSE_SAME = False
ACTIVATION_FUNCTIONS = ['relu', 'sigmoid', 'tanh']
DROPOUT_CHANCE = 0.7
MIN_DROPOUT = 0.1
MAX_DROPOUT = 0.3


def generate_model_config(output_activation='softmax'):
    model_config = {
        'model': 'General_RNN',
        'num_of_layers': np.random.randint(1, MAX_NUM_OF_LAYERS),
        'layers': [],
        "output_activation": output_activation
    }

    # Choose if adding dropout
    dropout = np.random.random()
    model_config['dropout'] = False if dropout > DROPOUT_CHANCE else True
    if model_config['dropout']:
        # choose dropout rate
        model_config['dropout_rate'] = np.random.uniform(MIN_DROPOUT, MAX_DROPOUT)

    # add activation function to Dense layers
    activation_function = np.random.choice(ACTIVATION_FUNCTIONS) if ALL_DENSE_SAME else None

    # crate model layers
    for layer in range(model_config['num_of_layers']):
        model_config['layers'].append({
            'type': np.random.choice(LAYERS_TYPES),
            'size': int(2 ** np.random.random_integers(MIN_SIZE, MAX_SIZE))
        })
        if model_config['layers'][layer]['type'] == 'Dense':
            model_config['layers'][layer]['activation_function'] = activation_function or np.random.choice(
                    ACTIVATION_FUNCTIONS)

    return model_config


def json_format(name, backup_config=None, loss_config=None, model_config=None, train_config=None,
                val_dataset_config=None, train_dataset_config=None):
    data_dic = {}
    data_dic['backup_config'] = backup_config or {
        "handler": "DefaultLocal",
        "history_backup_delta": 1,
        "project": "default_project",
        "save_history_after_training": True,
        "save_snapshot_after_training": True,
        "snapshot_backup_delta": 1
    }
    data_dic['loss_config'] = loss_config or {
        "loss": "categorical_crossentropy"
    }
    data_dic['model_config'] = model_config or {
        # "input_shape": [
        #     7,
        #     4
        # ],
        "model": "LSTM",
        "output_activation": "softmax",
        "output_shape": [
            2
        ],
        "units": 128
    }
    data_dic['name'] = name
    data_dic['train_config'] = train_config or {
        "augmentations": None,
        "batch_size": 1024,
        "callbacks": None,
        "epochs": 10,
        "include_experiment_callbacks": True,
        "optimizer": "adam",
        "randomize": True,
        "steps_per_epoch": None,
        "validation_steps": None
    }
    data_dic['train_dataset_config'] = train_dataset_config or {
        "config_path": None,
        "dataset": "StocksDataset",
        "feature_list": None,
        "output_feature_list": None,
        "stock_name_list": None,
        "time_sample_length": 7,
        "val_mode": False
    }
    data_dic['val_dataset_config'] = val_dataset_config or {
        "config_path": None,
        "dataset": "StocksDataset",
        "feature_list": None,
        "output_feature_list": None,
        "stock_name_list": None,
        "time_sample_length": 7,
        "val_mode": True
    }

    # json_data = json.dumps(data_dic)
    # return json_data
    return data_dic


# main
for i in range(3):
    model_name = 'RNN_MODEL_{test_number}'.format(test_number=i)
    file_name = r'.\Shmekel_Results\default_project\config_{name}.json'.format(name=model_name)

    model = {}
    model = generate_model_config(model_name)
    data = json_format(model_config=model, name=model_name)

    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)

exp1 = core.get_exp_from_config(
    core.load_config(r'.\Shmekel_Results\default_project\config_{name}.json'.format(name=model_name))
print(exp1.model.summary())
exp1.run()
