from loss_team_dir.pipe_line import get_model, get_data_generators, adjust_params, get_exp_name, compile_model, get_callbacks
from loss_team_dir.pipe_line import _default_loss_params as loss_kwargs
from loss_team_dir.pipe_line import _default_model_params as model_kwargs
import loss_team_dir.pipe_line.datasets as data
from loss_team_dir.pipe_line.post_training_confidence import get_intervals_accuracy


dataset = 'mnist'
noise_level = 2
batch_size = 1024
epochs = 50

adjust_params(loss_kwargs, model_kwargs)

model = get_model(**model_kwargs)
exp = get_exp_name(dataset_name=dataset, noise_level=noise_level,
                   loss_params={
                       'loss': 'categorical_crossentropy',
                       'hyper_parameters': 1,
                       'minimize': True,
                       'without_uncertainty': True
                   }, model_params=model_kwargs)

print('model_shapes: input', model.input_shape, ', output', model.output_shape)

compile_model(model, **loss_kwargs)

train_gen, val_gen = get_data_generators(dataset, batch_size=batch_size, noise_level=noise_level,
                                         num_classes=model_kwargs['output_shape'][-1])

steps_per_epoch = 60000 // batch_size
validation_steps = 10000 // batch_size
callbacks = get_callbacks()

h = model.fit_generator(train_gen, steps_per_epoch, epochs, verbose=2, callbacks=callbacks,
                        validation_data=val_gen, validation_steps=validation_steps)



###
num_classes = 10

(train_x, train_y), (val_x, val_y) = data.load_dataset(dataset)

train_x, val_x = data.normalize(train_x, val_x)

train_labels = data.get_labels(train_y, num_classes)
val_labels = data.get_labels(val_y, num_classes)

train_data = next(data._generator(train_x, 60000, labels=train_labels, randomize=True, noise_level=noise_level))
val_data = next(data._generator(val_x, 10000, labels=val_labels, randomize=True, noise_level=noise_level))


prob, cnt = get_intervals_accuracy(model, train_data[0], train_data[1], num_intervals=100)


