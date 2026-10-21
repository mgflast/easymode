"""Weights-only .h5 files in the Keras 2 layout, written from Keras 2 or Keras 3.

Keras 3 (TensorFlow >= 2.16) saves weights only as .weights.h5, which Keras 2 (TensorFlow <= 2.15)
cannot read, but its load_weights still reads the Keras 2 .h5 layout. So easymode writes that layout
on both, and a model trained on either loads in every easymode, released ones included.
"""
import numpy as np
import tensorflow as tf

# tf.keras.version() is new in Keras 3; Keras 2, tf-keras under TF_USE_LEGACY_KERAS=1 included, has none
KERAS3 = hasattr(tf.keras, 'version')
_HDF5_OBJECT_HEADER_LIMIT = 64512


def _save_attribute(group, name, data):
    # as Keras does: an attribute over the HDF5 header limit is split into name0, name1, ...
    chunks, n = [np.asarray(data)], 1
    while any(c.nbytes > _HDF5_OBJECT_HEADER_LIMIT for c in chunks):
        n += 1
        chunks = np.array_split(np.asarray(data), n)
    if n == 1:
        group.attrs[name] = data
    else:
        for i, chunk in enumerate(chunks):
            group.attrs[f'{name}{i}'] = chunk


def _save_group(group, weights):
    # Keras 3 names a variable by its last component ('kernel'); its path is unique, like a Keras 2 name
    names = [getattr(w, 'path', w.name).encode('utf8') for w in weights]
    _save_attribute(group, 'weight_names', names)
    for name, w in zip(names, weights):
        group.create_dataset(name, data=np.asarray(w))


def save_weights(model, path):
    """model.save_weights(path) to a .h5 file, in the layout Keras 2 writes."""
    if KERAS3:
        _write_keras2_layout(model, path)
    else:
        model.save_weights(path)


def _write_keras2_layout(model, path):
    import h5py
    with h5py.File(path, 'w') as f:
        _save_attribute(f, 'layer_names', [layer.name.encode('utf8') for layer in model.layers])
        f.attrs['backend'] = b'tensorflow'
        f.attrs['keras_version'] = tf.keras.version().encode('utf8')
        # one group per top-level layer, its weights trainable first: the order load_weights assigns them in
        for layer in sorted(model.layers, key=lambda l: l.name):
            _save_group(f.create_group(layer.name), layer.trainable_weights + layer.non_trainable_weights)
        in_layers = {id(w) for layer in model.layers for w in layer.weights}
        _save_group(f.create_group('top_level_model_weights'),
                    [w for w in model.trainable_weights + model.non_trainable_weights if id(w) not in in_layers])


class BestWeightsCheckpoint(tf.keras.callbacks.Callback):
    """ModelCheckpoint(path, monitor, save_best_only=True, save_weights_only=True, mode='min'), saving
    with save_weights() above: Keras 3's ModelCheckpoint writes only .weights.h5."""

    def __init__(self, path, monitor='loss', verbose=0):
        super().__init__()
        self.path = path
        self.monitor = monitor
        self.verbose = verbose
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None or not current < self.best:
            return
        if self.verbose:
            print(f'\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving model to {self.path}')
        self.best = current
        save_weights(self.model, self.path)
