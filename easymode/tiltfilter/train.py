import glob, os, mrcfile
import tensorflow as tf
import pandas as pd
import numpy as np
from itertools import count


GLOBAL_BINNING = 2
MIN_SAMPLES_PER_CLASS_PER_BATCH = 8
ROOT = '/cephfs/mlast/compu_projects/easymode'

class TrainingSample:
    idgen = count(0)

    def __init__(self, df_row):
        self.id = next(TrainingSample.idgen)
        self.dataset = df_row["dataset"]
        self.tiltseries = df_row["tiltseries"]
        self.reference_path = df_row["reference_path"]
        self.sample_path = df_row["sample_path"]
        self.image_width = df_row["image_width"]
        self.image_height = df_row["image_height"]
        self.image_label = int(df_row["image_label"])
        self.angle = df_row["angle"]
        self.transpose = False
        self.bucket = None

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

        j, k = x0.shape
        if j > k:
            x0 = x0.T
            x1 = x1.T
        return x0, x1, self.image_label

class DataLoader:
    def __init__(self, batch_size=32, validation=False):
        self.batch_size = batch_size
        self.validation = validation
        self.df = pd.read_parquet('/cephfs/mlast/compu_projects/easymode/training/tilt/database.parquet')
        self.samples = []
        self.buckets = {}
        # TODO: actually parse validation/training splits!
        self.parse_data()

        assert self.batch_size > 2 * MIN_SAMPLES_PER_CLASS_PER_BATCH, "Batch size should be at least 2 * MIN_SAMPLES_PER_CLASS_PER_BATCH to avoid class imbalance!"

    def parse_data(self):
        for _, row in self.df.iterrows():
            if row["image_label"] not in (0, 1):
                continue

            s = TrainingSample(row)
            self.samples.append(s)

        for s in self.samples:
            j, k = s.image_width, s.image_height
            if j > k:
                j, k = k, j
            if (j, k) in self.buckets:
                self.buckets[(j, k)].append(s)
                s.bucket = (j, k)
            else:
                self.buckets[(j, k)] = [s]
                s.bucket = (j, k)
                print(f'Created new bucket (n={len(self.buckets)}) with signature ({j}, {k}) (in DS {s.dataset})')

        for k in self.buckets:
            print(f'bucket {k} contains {len(self.buckets[k])} samples')

        n_total = len(self.samples)
        n_label_1 = len([f for f in self.samples if f.image_label == 1])
        n_label_0 = len([f for f in self.samples if f.image_label == 0])

        print(f'Training data contains:\n'
              f'{n_label_1} label-1 ({n_label_1/n_total*100.0:.1f}%)\n'
              f'{n_label_0} label-0 ({n_label_0/n_total*100.0:.1f}%)\n')


    def augment_batch(self, X0, X1, y):
        if self.validation:
            return X0, X1, y

        k = np.random.randint(0, 4)
        X0 = np.rot90(X0, k=k, axes=(1, 2))
        X1 = np.rot90(X1, k=k, axes=(1, 2))

        if np.random.rand() < 0.5:
            axis = np.random.randint(0, 2)
            X0 = np.flip(X0, axis=axis + 1)
            X1 = np.flip(X1, axis=axis + 1)

        return X0, X1, y


    def preprocess_batch(self, X0, X1, y):
        m0 = X0.mean(axis=(1, 2), keepdims=True)
        s0 = X0.std(axis=(1, 2), keepdims=True) + 1e-8
        X0 = (X0 - m0) / s0

        m1 = X1.mean(axis=(1, 2), keepdims=True)
        s1 = X1.std(axis=(1, 2), keepdims=True) + 1e-8
        X1 = (X1 - m1) / s1

        return X0, X1, y


    def batch_generator(self):
        while True:
            k = np.random.choice(self.samples).bucket  # choose a random sample, use it's bucket
            bucket = self.buckets[k]

            if len(bucket) < self.batch_size:
                continue

            label_1_samples = [f for f in bucket if f.image_label == 1]
            label_0_samples = [f for f in bucket if f.image_label == 0]

            n_label_1 = min(len(label_1_samples), MIN_SAMPLES_PER_CLASS_PER_BATCH)
            n_label_0 = min(len(label_0_samples), MIN_SAMPLES_PER_CLASS_PER_BATCH)

            samples = []
            samples += np.random.choice(label_1_samples, n_label_1, replace=False).tolist()
            samples += np.random.choice(label_0_samples, n_label_0, replace=False).tolist()

            remaining = self.batch_size - len(samples)
            if remaining > 0:
                all_remaining = [f for f in bucket if f not in samples]
                samples += np.random.choice(all_remaining, min(remaining, len(all_remaining)), replace=False).tolist()

            samples = samples[:self.batch_size]

            X0 = []
            X1 = []
            y  = np.empty((self.batch_size,), dtype=np.float32)

            for i, s in enumerate(samples):
                x0, x1, yi = s.get_data()
                X0.append(x0)
                X1.append(x1)
                y[i] = float(yi)

            X0 = np.stack(X0, axis=0).astype(np.float32)  # (B,H,W)
            X1 = np.stack(X1, axis=0).astype(np.float32)

            X0, X1, y = self.augment_batch(X0, X1, y)
            X0, X1, y = self.preprocess_batch(X0, X1, y)

            yield (X0[..., None], X1[..., None]), y[:, None]


    def as_generator(self):
        ds = tf.data.Dataset.from_generator(
            self.batch_generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(self.batch_size, None, None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, None, None, 1), dtype=tf.float32),
                ),
                tf.TensorSpec(shape=(self.batch_size, 1), dtype=tf.float32),
            ),
        )

        steps = len(self.samples) // self.batch_size
        return ds.prefetch(tf.data.AUTOTUNE), steps

def train_model(batch_size=32, epochs=100, lr_start=1e-3, lr_end=1e-5):
    from easymode.tiltfilter.model import create
    tf.config.run_functions_eagerly(False)

    print(f'Training tilt selection network')

    with tf.distribute.MirroredStrategy().scope():
        model = create()
    model.optimizer.learning_rate.assign(lr_start)
    # data loaders
    training_ds, training_steps = DataLoader(batch_size=batch_size).as_generator()
    validation_ds, validation_steps = None, None #, DataLoader(batch_size=batch_size, validation=True).as_generator()

    # callbacks
    checkpoint_dir = f'{ROOT}/training/tilt/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    # cb_checkpoint_val = tf.keras.callbacks.ModelCheckpoint(filepath=f'{checkpoint_dir}/' + "validation_loss",
    #                                                        monitor=f'val_loss',
    #                                                        save_best_only=True,
    #                                                        save_weights_only=True,
    #                                                        mode='min',
    #                                                        verbose=1)
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

    model.fit(training_ds, steps_per_epoch=training_steps, validation_data=validation_ds, validation_steps=validation_steps, epochs=epochs, validation_freq=5, callbacks=[cb_checkpoint_train, cb_lr, cb_csv], class_weight={0:10.0, 1:1.0})
