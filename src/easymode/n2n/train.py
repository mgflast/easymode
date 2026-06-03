"""
N2N-style supervised trainer for easymode denoising/dewedging models.

The sampling step (writing matched (x, y) box pairs to training/{mode}/volumes_*)
lives in `easymode.training.sampler`. This module owns the runtime dataloader and
the Keras `fit` loop.

Modes are defined in `easymode.training.sampler.MODES`. The deployed pairs are:

  mode='n2n'  -> (x=even, y=odd)     classic noise2noise on half-map pairs
  mode='ddw'  -> (x=raw,  y=ddw)     distillation from the per-dataset DDW2 teachers

Both produce weights with the same architecture (easymode.n2n.model.UNet) but on
different supervision, so any tomogram-shaped weights file is loadable by the
same `create()`.
"""
import glob, os
import mrcfile
import numpy as np
import tensorflow as tf
from easymode.segmentation.augmentations import *  # noqa: F401,F403 -- np/random shim used below
from easymode.training.sampler import ROOT, MODES


class N2NDataloader:
    """Runtime dataloader: reads boxes from training/{mode}/volumes_{split}/{x,y}/
    and yields augmented (x, y) pairs as a tf.data.Dataset.

    For modes whose two flavours are exchangeable (n2n: x=even, y=odd; either side
    is a valid n2n training target), we randomly swap x<->y during training. For
    asymmetric modes (ddw: x=raw, y=teacher-corrected), we never swap."""

    # modes where x and y are interchangeable noisy realisations of the same signal
    SYMMETRIC = {"n2n"}

    def __init__(self, mode, batch_size=32, box_size=96, validation=False):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; known: {sorted(MODES)}")
        self.mode = mode
        self.batch_size = batch_size
        self.box_size = box_size
        self.validation = validation
        self._dir = f"{ROOT}/training/{mode}/volumes_{'validation' if validation else 'training'}"
        # filenames are md5 hashes from the sampler -- just list them, no integer parse
        self.indices = sorted(os.path.splitext(os.path.basename(f))[0]
                              for f in glob.glob(f"{self._dir}/x/*.mrc"))

    @staticmethod
    def augment(x, y):
        k = np.random.randint(0, 4)
        x = np.rot90(x, k, axes=(1, 2))
        y = np.rot90(y, k, axes=(1, 2))
        if np.random.rand() < 0.5:
            x = np.rot90(x, k=2, axes=(0, 2))
            y = np.rot90(y, k=2, axes=(0, 2))
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=2)
            y = np.flip(y, axis=2)
        return x, y

    @staticmethod
    def preprocess(x, y):
        x = x.astype(np.float32)
        x -= np.mean(x)
        x /= np.std(x) + 1e-8
        y = y.astype(np.float32)
        y -= np.mean(y)
        y /= np.std(y) + 1e-8
        return x[..., None], y[..., None]

    def _load(self, idx):
        x = np.asarray(mrcfile.read(f"{self._dir}/x/{idx}.mrc"))
        y = np.asarray(mrcfile.read(f"{self._dir}/y/{idx}.mrc"))
        return x, y

    def sample_generator(self):
        while True:
            np.random.shuffle(self.indices)
            for i in self.indices:
                x, y = self._load(i)
                if not self.validation:
                    x, y = self.augment(x, y)
                x, y = self.preprocess(x, y)
                if (not self.validation) and self.mode in self.SYMMETRIC and np.random.rand() < 0.5:
                    yield y, x                    # n2n-style swap: target and input are interchangeable
                else:
                    yield x, y

    def as_dataset(self, batch_size, num_epochs=None):
        ds = tf.data.Dataset.from_generator(
            self.sample_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.box_size, self.box_size, self.box_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(self.box_size, self.box_size, self.box_size, 1), dtype=tf.float32),
            ),
        )
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        options.threading.private_threadpool_size = 32
        ds = ds.with_options(options)
        if num_epochs:
            ds = ds.repeat(num_epochs)
        ds = ds.batch(batch_size=batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds, len(self.indices) // batch_size


def train_n2n(mode, batch_size=32, box_size=96, epochs=100, lr_start=1e-3, lr_end=1e-5, temp=""):
    """Train the n2n-style UNet on the (x, y) pairs produced by the sampler for
    `mode`. Reads from training/{mode}/volumes_*/, writes checkpoints to
    training/{mode}/checkpoints/ (or `temp` if provided)."""
    from easymode.n2n.model import create

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; known: {sorted(MODES)}")
    tf.config.run_functions_eagerly(False)
    print(f"\nTraining n2n-style model for mode: {mode} "
          f"(x={MODES[mode][0]}, y={MODES[mode][1]})\n")

    with tf.distribute.MirroredStrategy().scope():
        model = create()
    model.optimizer.learning_rate.assign(lr_start)

    train_ds, train_steps = N2NDataloader(mode=mode, batch_size=batch_size, box_size=box_size,
                                          validation=False).as_dataset(batch_size=batch_size)
    val_ds, val_steps = N2NDataloader(mode=mode, batch_size=batch_size, box_size=box_size,
                                      validation=True).as_dataset(batch_size=batch_size)

    ckpt_dir = temp if temp else f"{ROOT}/training/{mode}/checkpoints/"
    if not ckpt_dir.endswith("/"):
        ckpt_dir += "/"
    os.makedirs(ckpt_dir, exist_ok=True)

    cb_ckpt_train = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{ckpt_dir}training_loss",
        monitor="loss", save_best_only=True, save_weights_only=True, mode="min", verbose=1,
    )
    cb_ckpt_val = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{ckpt_dir}validation_loss",
        monitor="val_loss", save_best_only=True, save_weights_only=True, mode="min", verbose=1,
    )

    def lr_decay(epoch, _):
        return float(lr_start + (lr_end - lr_start) * ((epoch - 2) / epochs))
    cb_lr = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=1)

    cb_csv = tf.keras.callbacks.CSVLogger(f"{ckpt_dir}training_log.csv", append=True)

    model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        epochs=epochs,
        validation_freq=1,
        callbacks=[cb_ckpt_val, cb_ckpt_train, cb_lr, cb_csv],
    )
