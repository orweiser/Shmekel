from keras import Model
from keras.layers import Dense, Input, Flatten
from keras.layers.normalization import BatchNormalization
from read_data import data_reader
from preprocess_functions import *
import matplotlib.pyplot as plt


class Perceptron:
    """
    don't pay too much attention to this class, to try some models use the function "perceptron" instead
    """
    def __init__(self, input_shape, output_shape=(2,), width=128, num_hidden_layers=3, activation='relu',
                 loss='categorical_crossentropy', metrics='acc', last_layer_activation='softmax', opt='adam',
                 batch_size=512, use_BN=True):

        if metrics is not None:
            if type(metrics) is not list:
                metrics = [metrics]

        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.width = width
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.loss = loss
        self.metrics = metrics
        self.last_layer_activation = last_layer_activation
        self.use_BN = use_BN
        self.opt = opt

        self.units_per_hidden_layer = self.__get_units_per_hidden_layer()

        self.model = self.__get_a_model()
        self.__compile()

    def __get_units_per_hidden_layer(self):
        if type(self.width) is list:
            if len(self.width) != self.num_hidden_layers:
                print('Warning: parameter num_hidden_layers does not match len(width). ignoring num_hidden_layers')
            return self.width

        return [self.width] * self.num_hidden_layers

    def __get_a_model(self):
        input_layer = Input(shape=self.input_shape, name='input_layer')

        # x = Flatten()(input_layer)
        x = input_layer

        for i, units in enumerate(self.units_per_hidden_layer):
            x = Dense(units, activation=self.activation, name='hidden_layer_' + str(i+1))(x)

            if self.use_BN and (i + 1) < len(self.units_per_hidden_layer):
                x = BatchNormalization(name='normalization_layer_' + str(i+1))(x)

        num_output_units = np.prod(self.output_shape)
        x = Dense(num_output_units, activation=self.last_layer_activation, name='output_layer')(x)

        return Model(input_layer, x, name='perceptron_model')

    def __compile(self):
        self.model.compile(loss=self.loss, optimizer=self.opt, metrics=self.metrics)


def perceptron(input_shape, output_shape=(2,), width=128, num_hidden_layers=3, activation='relu',
               loss='categorical_crossentropy', metrics='acc', last_layer_activation='softmax', opt='adam',
               batch_size=512, use_BN=True):
    """
    this function creates and compiles a simple perceptron (fully connected) model.
    :param input_shape: the inout shape of the data as a tuple. do not include batch size
    :param output_shape: the shape of the labels & predictions as a tuple without the batch_size
    :param width: an integer or a list of integers to determine the number of units in each hidden layer
    :param num_hidden_layers: only relevant if width is an integer
    :param activation: the hidden_layers activation.
    :param loss:
        for regression use 'mean_squared_error'
        for binary classification us 'binary_crossentropy'
        for categorical classification us 'categorical_crossentropy'
    :param metrics: a list of keras metrics
    :param last_layer_activation: 'sigmoid' for binary classification, 'softmax' for categorical classification
    :param opt: a keras optimzer
    :param batch_size:
    :param use_BN: boolean if to use normalization layers
    :return: a keras model
    """
    return Perceptron(**locals()).model


# A stupid example: lets try to let a model learn if there have been a rise in the stock value today from todays data:
# we expect to succeed because no prediction need to be made

# get a data array:
train_data = next(data_reader(min_time_stamps_per_samples=10000, randomize=False))
validation_data = next(data_reader(min_time_stamps_per_samples=10000, randomize=False))

# do preprocess on data:
train_inputs = normalize_data_array(train_data)
validation_inputs = normalize_data_array(validation_data)

# make labels:
train_labels = feature_difference(train_data, keys=('Close', 'Open'), bool_type='categorical')
validation_labels = feature_difference(validation_data, keys=('Close', 'Open'), bool_type='categorical')

# now we got some data
# the model will train according to the training data and labels and the evaluation is done on the validation

# create a model:
model = perceptron(
    input_shape=train_data.shape[1:],  # input shape is the shape of the data without the number of samples
    output_shape=train_labels.shape[1:],  # output shape is the shape of the labels without the number of samples
    width=[16, 32, 16],  # this says that we want a model with 3 hidden layers with units as stated
)

# lets check out the model to see if it is as we expected:
model.summary()

# everything seems fine, so lets start training our model:
history = model.fit(x=train_inputs, y=train_labels, batch_size=128, epochs=20, verbose=2)

# now the model is trained. let us compare the performances on train data vs validation data:
train_performances = model.evaluate(x=train_inputs, y=train_labels, batch_size=1024)
validation_performances = model.evaluate(x=validation_inputs, y=validation_labels, batch_size=128)

print('Test performances:')
print(model.metrics_names)
print(train_performances)

print('Validation performances:')
print(model.metrics_names)
print(validation_performances)

# actually, we don't wait until the end of the training to check our performances on validation set:
# history = model.fit(x=train_inputs, y=train_labels, batch_size=128, epochs=10, verbose=2,
#                     validation_data=(validation_inputs, validation_labels))

# how to use the history stuff?
history = history.history  # just the way to do it, no real explanation here

plt.figure()
legend = []
for metric_name, values in history.items():
    plt.plot(values)
    legend.append(metric_name)

plt.legend(legend)
plt.grid()
plt.xlabel('# epochs')

