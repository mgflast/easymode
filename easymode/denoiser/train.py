import glob, os, mrcfile, numpy as np, random
from easymode.core.augmentations import *
import tensorflow as tf

BOX_SIZE = 128


class DataLoader:
    def __init__(self, batch_size=8, validation=False, flavour='cryocare'):
        print(f'Initializing DataLoader for easymode denoise with flavour {flavour}.')
        self.batch_size = batch_size
        self.validation = validation
        self.flavour = flavour
        self.volume_pairs = list()

    def load_data(self):
        all_tomograms = [os.path.basename(f) for f in glob.glob(f'/cephfs/mlast/compu_projects/easymode/volumes_{self.flavour}/*.mrc')]
        for tomo in all_tomograms:
            path_volume_x = f'/cephfs/mlast/compu_projects/easymode/volumes_raw/{tomo}'
            path_volume_y = f'/cephfs/mlast/compu_projects/easymode/volumes_{self.flavour}/{tomo}'
            if os.path.exists(path_volume_x) and os.path.exists(path_volume_y):
                self.volume_pairs.append((path_volume_x, path_volume_y))

        print(f'Loaded {len(self.volume_pairs)} volume pairs.')

    @staticmethod
    def augment(train_x, train_y, validation=False):
        if validation:
            return train_x, train_y

        k = np.random.randint(0, 4)
        train_x = np.rot90(train_x, k, axes=(1, 2))
        train_y = np.rot90(train_y, k, axes=(1, 2))

        if np.random.rand() < 0.5:
            train_x = np.rot90(train_x, k, axes=(0, 2))
            train_y = np.rot90(train_y, k, axes=(0, 2))

        if np.random.rand() < 0.5:
            train_x = np.flip(train_x, axis=2)
            train_y = np.flip(train_y, axis=2)

        return train_x, train_y

    @staticmethod
    def preprocess(train_x, train_y):
        train_x = train_x.astype(np.float32)
        train_x -= np.mean(train_x)
        train_x /= np.std(train_x) + 1e-8

        train_y = train_y.astype(np.float32)
        train_y -= np.mean(train_y)
        train_y /= np.std(train_y) + 1e-8

        train_x = np.expand_dims(train_x, axis=-1)
        train_y = np.expand_dims(train_y, axis=-1)

        return train_x, train_y

    @staticmethod
    def get_sample(volume_x, volume_y):
        volume_shape = mrcfile.mmap(volume_x).data.shape

        box_start = [random.randint(0, volume_shape[i] - BOX_SIZE) for i in range(3)]
        if volume_shape[0] > 2 * BOX_SIZE:
            box_start[0] = volume_shape[0] // 2 + random.randint(-BOX_SIZE, BOX_SIZE)

        train_x = mrcfile.mmap(volume_x).data[box_start[0]:box_start[0] + BOX_SIZE, box_start[1]:box_start[1] + BOX_SIZE, box_start[2]:box_start[2] + BOX_SIZE].copy()
        train_y = mrcfile.mmap(volume_y).data[box_start[0]:box_start[0] + BOX_SIZE, box_start[1]:box_start[1] + BOX_SIZE, box_start[2]:box_start[2] + BOX_SIZE].copy()

        return train_x, train_y

    def sample_generator(self):
        while True:
            np.random.shuffle(self.volume_pairs)
            for volume_x, volume_y in self.volume_pairs:
                train_x, train_y = self.get_sample(volume_x, volume_y)
                train_x, train_y = self.augment(train_x, train_y, self.validation)
                train_x, train_y = self.preprocess(train_x, train_y)
                yield train_x, train_y

    def as_generator(self, batch_size):
        dataset = tf.data.Dataset.from_generator(self.sample_generator, output_signature=(tf.TensorSpec(shape=(160, 160, 160, 1), dtype=tf.float32), tf.TensorSpec(shape=(160, 160, 160, 1), dtype=tf.float32))).batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        n_steps = len(self.samples) // batch_size
        return dataset, n_steps


def train_model(title='', features='', batch_size=8, epochs=2000, lr_start=1e-3, lr_end=1e-5):
    from easymode.core.model import create

    tf.config.run_functions_eagerly(False)

    print(f'\nTraining model with features: {features}\n')

    with tf.distribute.MirroredStrategy().scope():
        model = create()
    model.optimizer.learning_rate.assign(lr_start)

    # data loaders
    training_ds, training_steps = DataLoader(features, batch_size=batch_size, validation=False).as_generator(batch_size=batch_size)
    validation_ds, validation_steps = DataLoader(features, batch_size=batch_size, validation=True).as_generator(batch_size=batch_size)

    # callbacks
    os.makedirs(f'/cephfs/mlast/compu_projects/easymode/training/3d/checkpoints/{title}', exist_ok=True)
    cb_checkpoint_val = tf.keras.callbacks.ModelCheckpoint(filepath=f'/cephfs/mlast/compu_projects/easymode/training/3d/checkpoints/{title}/' + "validation_loss",
                                                           monitor=f'val_loss',
                                                           save_best_only=True,
                                                           save_weights_only=True,
                                                           mode='min',
                                                           verbose=1)
    cb_checkpoint_train = tf.keras.callbacks.ModelCheckpoint(filepath=f'/cephfs/mlast/compu_projects/easymode/training/3d/checkpoints/{title}/' + "training_loss",
                                                             monitor=f'loss',
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode='min',
                                                             verbose=1)

    def lr_decay(epoch, _):
        return float(lr_start + (lr_end - lr_start) * ((epoch - 2) / epochs))

    cb_lr = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=1)

    cb_csv = tf.keras.callbacks.CSVLogger(f'/cephfs/mlast/compu_projects/easymode/training/3d/checkpoints/{title}/training_log.csv', append=True)
    model.fit(training_ds, steps_per_epoch=training_steps, validation_data=validation_ds, validation_steps=validation_steps, epochs=500, validation_freq=1, callbacks=[cb_checkpoint_val, cb_checkpoint_train, cb_lr, cb_csv])
