import glob, os, mrcfile
import tensorflow as tf
import pandas as pd
import numpy as np

GLOBAL_BINNING = 2
ROOT = '/cephfs/mlast/compu_projects/easymode'

class TrainingSample:
    def __init__(self, df_row):
        self.dataset = df_row["dataset"]
        self.tiltseries = df_row["tiltseries"]
        self.reference_path = df_row["reference_path"]
        self.sample_path = df_row["sample_path"]
        self.image_width = df_row["image_width"]
        self.image_height = df_row["image_height"]
        self.image_label = int(df_row["image_label"])
        self.angle = df_row["angle"]

    @staticmethod
    def bin(x):
        b = GLOBAL_BINNING
        j, k = x.shape
        x = x[:j//b*b, :k//b*b]
        x = x.reshape(j//b, b, k//b, b).mean(3).mean(1)
        return x

    def get_data(self):
        x0_path = f'/cephfs/mlast/compu_projects/easymode/training/tilt/data/datasets/{self.dataset}/warp_frameseries/average/{os.path.basename(self.reference_path)}'
        x1_path = f'/cephfs/mlast/compu_projects/easymode/training/tilt/data/datasets/{self.dataset}/warp_frameseries/average/{os.path.basename(self.sample_path)}'
        x0 = TrainingSample.bin(mrcfile.read(x0_path).astype(np.float32))
        x1 = TrainingSample.bin(mrcfile.read(x1_path).astype(np.float32))
        return x0, x1, self.image_label

class DataLoader:
    def __init__(self, batch_size=32, validation=False):
        self.batch_size = batch_size
        self.validation = validation
        self.df = pd.read_parquet('/cephfs/mlast/compu_projects/easymode/training/tilt/database.parquet')
        self.samples = []

        self.parse_data()

    def parse_data(self):
        for _, row in self.df.iterrows():
            if row["image_label"] not in (0, 1):
                continue

            s = TrainingSample(row)
            self.samples.append(s)

    def augment(self, x0, x1, y):
        if self.validation:
            return x0, x1, y

        k = np.random.randint(0, 4)
        x0 = np.rot90(x0, k=k, axes=(0, 1))
        x1 = np.rot90(x1, k=k, axes=(0, 1))

        if np.random.uniform(0.0, 1.0) < 0.5:
            k = np.random.randint(0, 1)
            x0 = np.flip(x0,axis=k)
            x1 = np.flip(x1, axis=k)

        return x0, x1, y

    def preprocess(self, x0, x1, y):
        x0 -= np.mean(x0)
        x0 /= np.std(x0) + 1e-8
        x1 -= np.mean(x1)
        x1 /= np.std(x1) + 1e-8

        return x0, x1, y

    def sample_generator(self):
        while True:
            np.random.shuffle(self.samples)
            for s in self.samples:
                x0, x1, y = s.get_data()
                if not self.validation:
                    x0, x1, y = self.augment(x0, x1, y)
                x0, x1, y = self.preprocess(x0, x1, y)

                yield x0, x1, y

    def as_generator(self):
        dataset = tf.data.Dataset.from_generator(
            self.sample_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )

        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=(
                (None, None),
                (None, None),
                (),
            ),
            padding_values=(0.0, 0.0, 0),
            drop_remainder=True,
        )

        dataset = dataset.map(
            lambda x0, x1, y: (
                (x0[..., None], x1[..., None]),  # add channel dim
                tf.cast(y, tf.float32)[..., None]
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        return dataset.prefetch(tf.data.AUTOTUNE), len(self.samples) // self.batch_size

def train_model(batch_size=32, epochs=100, lr_start=1e-3, lr_end=1e-5):
    from easymode.tiltfilter.model import create
    tf.config.run_functions_eagerly(False)

    print(f'Training tilt selection network')

    with tf.distribute.MirroredStrategy().scope():
        model = create()
    model.optimizer.learning_rate.assign(lr_start)
    # data loaders
    training_ds, training_steps = DataLoader(batch_size=batch_size).as_generator()
    validation_ds, validation_steps = DataLoader(batch_size=batch_size, validation=True).as_generator()

    # callbacks
    checkpoint_dir = f'{ROOT}/training/tilt/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    cb_checkpoint_val = tf.keras.callbacks.ModelCheckpoint(filepath=f'{checkpoint_dir}/' + "validation_loss",
                                                           monitor=f'val_loss',
                                                           save_best_only=True,
                                                           save_weights_only=True,
                                                           mode='min',
                                                           verbose=1)
    cb_checkpoint_train = tf.keras.callbacks.ModelCheckpoint(filepath=f'{checkpoint_dir}/' + "training_loss",
                                                             monitor=f'loss',
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode='min',
                                                             verbose=1)

    def lr_decay(epoch, _):
        return float(lr_start + (lr_end - lr_start) * (epoch / epochs))

    cb_lr = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=1)

    cb_csv = tf.keras.callbacks.CSVLogger(f'{checkpoint_dir}/training_log.csv', append=True)

    model.fit(training_ds, steps_per_epoch=training_steps, validation_data=validation_ds, validation_steps=validation_steps, epochs=epochs, validation_freq=5, callbacks=[cb_checkpoint_val, cb_checkpoint_train, cb_lr, cb_csv])
