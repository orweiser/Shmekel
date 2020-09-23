from keras.callbacks import Callback
from keras import callbacks as keras_callbacks
from keras.callbacks import TensorBoard


def _parse_item(item):
    if isinstance(item, str):
        return item, {}
    if isinstance(item, dict):
        return item['name'], {key: val for key, val in item.items() if key != 'name'}
    if isinstance(item, (list, tuple)):
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], dict)
        return tuple(item)


def parse_callback(item):
    if isinstance(item, (keras_callbacks.Callback, TensorBoard)):
        return item

    c_name, c_params = _parse_item(item)

    try:
        module = globals()[c_name]
    except KeyError:
        module = getattr(keras_callbacks, c_name)
    return module(**c_params)


class DebugCallback(Callback):
    def __init__(self):
        super(DebugCallback, self).__init__()
        self.was_called = False

    def on_train_begin(self, logs=None):
        self.was_called = True

    def on_batch_end(self, batch, logs=None):
        self.model.stop_training = True
