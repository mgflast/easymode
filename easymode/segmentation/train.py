import glob, os, mrcfile
from easymode.segmentation.augmentations import *
import tensorflow as tf

AUGMENTATIONS_ROT_XZ_YZ = 0.5
AUGMENTATIONS_ROT_XY = 0.5
AUGMENTATIONS_MISSING_WEDGE = 0.25
AUGMENTATIONS_GAUSSIAN = 0.33
AUGMENTATIONS_SCALE = 0.33
AUGMENTATIONS_MIXUP = 0.2

DEBUG = False
ROOT = '/cephfs/mlast/compu_projects/easymode'
if os.name == 'nt':
    DEBUG = True
    ROOT = 'Z:/compu_projects/easymode'
    print(f'debug mode - running on Windows with root at {ROOT}')


class Sample:
    FLAVOURS = ['odd', 'even', 'raw', 'cryocare']
    def __init__(self, idx, datagroup):
        self.idx = idx
        self.datagroup = datagroup
        self.valid = True
        self.is_positive = False
        self.flavours = {}
        self.label_path = f'{ROOT}/training/3d/data/{self.datagroup}/label/{self.idx}.mrc'
        self.validity_path = f'{ROOT}/training/3d/data/{self.datagroup}/validity/{self.idx}.mrc'

        # check existence of label volume
        if not os.path.exists(self.label_path):
            print(f'Missing label for sample {self.idx} in {self.datagroup}')
            self.valid = False
        else:
            label_volume = mrcfile.read(self.label_path)
            self.is_positive = np.sum(label_volume == 1) > 0

        # check existence of validity volume
        if not os.path.exists(self.validity_path):
            print(f'Missing validity for sample {self.idx} in {self.datagroup}')
            self.valid = False

        # check existence of subtomogram flavours
        for f in Sample.FLAVOURS:
            volume_path = f'{ROOT}/training/3d/data/{self.datagroup}/{f}/{self.idx}.mrc'
            if os.path.exists(volume_path):
                self.flavours[f] = volume_path
            else:
                print(f'Missing flavour {f} for sample {self.idx} in {self.datagroup}')

        if len(self.flavours) == 0:
            print(f'No available flavours for sample {self.idx} in {self.datagroup}')
            self.valid = False

    def load(self):
        available_flavours = list(self.flavours.keys())
        if 'cryocare' in available_flavours:
            available_flavours.append('cryocare')  # double odds of selecting cryocare
        flavours = random.sample(available_flavours, 2)
        mixing_factor = random.uniform(0.0, 1.0)

        img_a = mrcfile.read(self.flavours[flavours[0]])
        img_b = mrcfile.read(self.flavours[flavours[1]])
        img = img_a * mixing_factor + img_b * (1 - mixing_factor)

        label = mrcfile.read(self.label_path)
        validity = mrcfile.read(self.validity_path)

        label[validity == 0] = 2
        label[:16, :, :] = 2
        label[-16:, :, :] = 2
        label[:, :16, :] = 2
        label[:, -16:, :] = 2
        label[:, :, :16] = 2
        label[:, :, -16:] = 2

        if self.datagroup == 'Junk3D' or self.datagroup.startswith('Not'):
            label[label == 1] = 0

        return img, label


class DataLoader:
    def __init__(self, features, batch_size=8, validation=False, limit_z=False):
        self.features = features
        self.batch_size = batch_size
        self.validation = validation
        self.limit_z = limit_z
        self.samples = list()
        self.positive_samples = list()
        self.negative_samples = list()
        self.load_data()

    def load_data(self):
        self.samples = list()
        for f in self.features:
            available_samples = [os.path.basename(n).split('.')[0] for n in glob.glob(f'{ROOT}/training/3d/data/{f}/raw/*.mrc')]
            for n in available_samples:
                sample = Sample(idx=n, datagroup=f)
                if sample.valid:
                    self.samples.append(sample)

        if self.validation:
            self.samples = [s for i, s in enumerate(self.samples) if i % 20 == 0]
        else:
            self.samples = [s for i, s in enumerate(self.samples) if i % 20 != 0]

        self.positive_samples = [s for s in self.samples if s.is_positive]
        self.negative_samples = [s for s in self.samples if not s.is_positive]
        np.random.shuffle(self.samples)

        print(f'Loaded {len(self.samples)} samples for {"validation" if self.validation else "training"}')

    def mixup(self, img, label):
        negative_sample = random.choice(self.negative_samples)
        negative_img, _ = negative_sample.load()
        mixing_factor = random.uniform(0.0, 0.5)

        img = img * (1 - mixing_factor) + negative_img * mixing_factor
        return img, label


    def augment(self, img, label):
        if self.validation:
            return img, label

        # AUGMENTATION 1 - 0, 90, 180, or 270 degree rotation around Z axis.
        img, label = rotate_90_xy(img, label)

        # AUGMENTATION 2 - 0 or 180 degree rotation around Y axis.
        img, label = rotate_90_xz(img, label)

        # AUGMENTATION 3 - random flip along any axis. Note that this breaks chirality; but if anything this will help when tomograms have the wrong handedness.
        img, label = flip(img, label)

        # AUGMENTATION 4 - Gaussian filtering
        if random.uniform(0.0, 1.0) < AUGMENTATIONS_GAUSSIAN:
            img, label = filter_gaussian(img, label)

        # AUGMENTATION 5 - rotate by a random angle between -20 and +20 degrees around X or Y axis
        if random.uniform(0.0, 1.0) < AUGMENTATIONS_ROT_XZ_YZ:
            img, label = rotate_continuous_xz_or_yz(img, label)

        # AUGMENTATION 6 - rotate randomly along Z axis
        if random.uniform(0.0, 1.0) < AUGMENTATIONS_ROT_XY:
            img, label = rotate_continuous_xy(img, label)

        # AUGMENTATION 7 - missing wedge simulation
        if random.uniform(0.0, 1.0) < AUGMENTATIONS_MISSING_WEDGE:
            img, label = remove_wedge(img, label)

        # AUGMENTATION 8 - magnification jitter (90% to 110%)
        if random.uniform(0.0, 1.0) < AUGMENTATIONS_SCALE:
            img, label = scale(img, label)

        # AUGMENTATION 9 - mixup between the sample and a random negative sample
        if random.uniform(0.0, 1.0) < AUGMENTATIONS_MIXUP:
            img, label = self.mixup(img, label)

        label[:16, :, :] = 2
        label[-16:, :, :] = 2
        label[:, :16, :] = 2
        label[:, -16:, :] = 2
        label[:, :, :16] = 2
        label[:, :, -16:] = 2

        return img, label

    def preprocess(self, img, label):
        img = img.astype(np.float32)
        img = img - np.mean(img)
        img /= np.std(img) + 1e-8

        label = label.astype(np.float32)

        img = np.expand_dims(img, axis=-1)
        label = np.expand_dims(label, axis=-1)

        return img, label

    def sample_generator(self):
        while True:
            np.random.shuffle(self.samples)
            for j in range(len(self.samples)):
                if j % 4 == 0:
                    sample = random.choice(self.positive_samples)
                else:
                    sample = self.samples[j]

                img, label = sample.load()
                if not self.validation:
                    img, label = self.augment(img, label)
                img, label = self.preprocess(img, label)
                if self.limit_z:
                    img = img[32:128, :, :]
                    label = label[32:128, :, :]
                yield img, label

    def as_generator(self, batch_size):
        z_dim = 96 if self.limit_z else 160
        dataset = tf.data.Dataset.from_generator(self.sample_generator, output_signature=(tf.TensorSpec(shape=(z_dim, 160, 160, 1), dtype=tf.float32), tf.TensorSpec(shape=(z_dim, 160, 160, 1), dtype=tf.float32))).batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        n_steps = len(self.samples) // batch_size
        return dataset, n_steps


def train_model(title='', features='', batch_size=8, epochs=100, lr_start=1e-3, lr_end=1e-5, architecture_version=1, limit_z=False):
    # Import appropriate model architecture
    if architecture_version == 2:
        from easymode.segmentation.model_v2 import create
        print(f'\nTraining model v2 (new architecture) with features: {features}\n')
    else:
        from easymode.segmentation.model import create
        print(f'\nTraining model v1 (original architecture) with features: {features}\n')

    if limit_z:
        print('Limiting Z dimension to central 96 voxels.\n')

    tf.config.run_functions_eagerly(False)

    print(f'\nTraining model with features: {features}\n')

    with tf.distribute.MirroredStrategy().scope():
        model = create()
    model.optimizer.learning_rate.assign(lr_start)

    # data loaders
    training_ds, training_steps = DataLoader(features, batch_size=batch_size, validation=False, limit_z=limit_z).as_generator(batch_size=batch_size)
    validation_ds, validation_steps = DataLoader(features, batch_size=batch_size, validation=True, limit_z=limit_z).as_generator(batch_size=batch_size)

    # callbacks
    os.makedirs(f'{ROOT}/training/3d/checkpoints/{title}', exist_ok=True)
    cb_checkpoint_val = tf.keras.callbacks.ModelCheckpoint(filepath=f'{ROOT}/training/3d/checkpoints/{title}/' + "validation_loss",
                                                           monitor=f'val_loss',
                                                           save_best_only=True,
                                                           save_weights_only=True,
                                                           mode='min',
                                                           verbose=1)
    cb_checkpoint_train = tf.keras.callbacks.ModelCheckpoint(filepath=f'{ROOT}/training/3d/checkpoints/{title}/' + "training_loss",
                                                             monitor=f'loss',
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode='min',
                                                             verbose=1)

    def lr_decay(epoch, _):
        return float(lr_start + (lr_end - lr_start) * (epoch / epochs))

    cb_lr = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=1)

    cb_csv = tf.keras.callbacks.CSVLogger(f'{ROOT}/training/3d/checkpoints/{title}/training_log.csv', append=True)
    model.fit(training_ds, steps_per_epoch=training_steps, validation_data=validation_ds, validation_steps=validation_steps, epochs=epochs, validation_freq=5, callbacks=[cb_checkpoint_val, cb_checkpoint_train, cb_lr, cb_csv])

