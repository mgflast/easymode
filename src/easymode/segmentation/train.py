import glob, os, mrcfile
from easymode.segmentation.augmentations import *
import tensorflow as tf

AUGMENTATIONS_ROT_XZ_YZ = 0.5 #0.33 #0.333
AUGMENTATIONS_ROT_XY = 0.5 #0.33 #0.333
AUGMENTATIONS_MISSING_WEDGE = 0.0 # 0.0
AUGMENTATIONS_GAUSSIAN = 0.5 #0.33 #0.2
AUGMENTATIONS_SCALE = 0.33 #0.33


DEBUG = False
ROOT = '/cephfs/mlast/compu_projects/easymode'
if os.name == 'nt':
    DEBUG = True
    ROOT = 'Z:/compu_projects/easymode'
    print(f'debug mode - running on Windows with root at {ROOT}')

class DataLoader:
    def __init__(self, features, batch_size=8, validation=False):
        self.features = features
        self.batch_size = batch_size
        self.validation = validation
        self.samples = list()
        self.positive_samples = list()
        self.load_data()

    def load_data(self):
        self.samples = list()
        for f in self.features:
            available_samples = [os.path.basename(n).split('.')[0] for n in glob.glob(f'{ROOT}/training/3d/data/{f}/raw/*.mrc')]
            for n in available_samples:
                self.samples.append((f, n))

        np.random.shuffle(self.samples)

        for j in range(len(self.samples)):
            if self.check_label_positivity(j):
                self.positive_samples.append(self.samples[j])

        if self.validation:
            self.samples = [s for i, s in enumerate(self.samples) if i % 20 == 0]
        else:
            self.samples = [s for i, s in enumerate(self.samples) if i % 20 != 0]

        print(f'Loaded {len(self.samples)} samples for {"validation" if self.validation else "training"}')

    def get_sample(self, datagroup, index):
        flavours = random.sample(['even', 'odd', 'cryocare', 'cryocare', 'raw', 'cryocare'], 2) ## 251127: test training predominantly on lowest-noise flavours
        mixing_factor = random.uniform(0.0, 1.0)

        img_a = mrcfile.read(f'{ROOT}/training/3d/data/{datagroup}/{flavours[0]}/{index}.mrc')
        img_b = mrcfile.read(f'{ROOT}/training/3d/data/{datagroup}/{flavours[1]}/{index}.mrc')
        img = img_a * mixing_factor + img_b * (1 - mixing_factor)

        label = mrcfile.read(f'{ROOT}/training/3d/data/{datagroup}/label/{index}.mrc')
        validity = mrcfile.read(f'{ROOT}/training/3d/data/{datagroup}/validity/{index}.mrc')  # binary mask of valid regions (1 = valid, 0 = invalid)

        label[validity == 0] = 2 ## training output: 0 = background, 1 = feature, 2 = ignore
        label[:16, :, :] = 2
        label[-16:, :, :] = 2
        label[:, :16, :] = 2
        label[:, -16:, :] = 2
        label[:, :, :16] = 2
        label[:, :, -16:] = 2

        if datagroup == 'Junk3D' or 'Not' in datagroup:
            label[label == 1] = 0

        return img, label

    def check_label_positivity(self, idx):
        sample = self.samples[idx]
        label = mrcfile.read(f'{ROOT}/training/3d/data/{sample[0]}/label/{sample[1]}.mrc')
        return np.sum(label == 1) > 0

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

        # AUGMENTATION 7 - missing wedge simulation - exclusive
        if random.uniform(0.0, 1.0) < AUGMENTATIONS_MISSING_WEDGE:
            img, label = remove_wedge(img, label)

        if random.uniform(0.0, 1.0) < AUGMENTATIONS_SCALE:
            img, label = scale(img, label)

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
            np.random.shuffle(self.positive_samples)
            for j in range(len(self.samples)):
                if j % self.batch_size == 0:
                    datagroup, index = self.positive_samples[(j // self.batch_size) % len(self.positive_samples)]
                else:
                    datagroup, index = self.samples[j]


                img, label = self.get_sample(datagroup, index)
                if not self.validation:
                    img, label = self.augment(img, label)
                img, label = self.preprocess(img, label)
                yield img, label

    def as_generator(self, batch_size):
        dataset = tf.data.Dataset.from_generator(self.sample_generator, output_signature=(tf.TensorSpec(shape=(160, 160, 160, 1), dtype=tf.float32), tf.TensorSpec(shape=(160, 160, 160, 1), dtype=tf.float32))).batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        n_steps = len(self.samples) // batch_size
        return dataset, n_steps


def train_model(title='', features='', batch_size=8, epochs=100, lr_start=1e-3, lr_end=1e-5):
    from easymode.segmentation.model import create

    tf.config.run_functions_eagerly(False)

    print(f'\nTraining model with features: {features}\n')

    with tf.distribute.MirroredStrategy().scope():
        model = create()
    model.optimizer.learning_rate.assign(lr_start)

    # data loaders
    training_ds, training_steps = DataLoader(features, batch_size=batch_size, validation=False).as_generator(batch_size=batch_size)
    validation_ds, validation_steps = DataLoader(features, batch_size=batch_size, validation=True).as_generator(batch_size=batch_size)

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
    model.fit(training_ds, steps_per_epoch=training_steps   , validation_data=validation_ds, validation_steps=validation_steps, epochs=epochs, validation_freq=1, callbacks=[cb_checkpoint_val, cb_checkpoint_train, cb_lr, cb_csv])


if __name__ == "__main__":
    features = ['NotMitochondrion3D']
    loader = DataLoader(features, batch_size=8, validation=False)

    os.makedirs('C:/Users/Mart Last/Desktop/debug', exist_ok=True)

    for k in range(8):
        datagroup, index = loader.samples[0]
        img, label = loader.get_sample(datagroup, index)
        # img, label = loader.augment(img, label)
        # img, label = loader.preprocess(img, label)

        img_squeeze = np.squeeze(img)
        label_squeeze = np.squeeze(label)

        mrcfile.write(f'C:/Users/Mart Last/Desktop/debug/test_img_{k}.mrc', img_squeeze, overwrite=True)
        mrcfile.write(f'C:/Users/Mart Last/Desktop/debug/test_label_{k}.mrc', label_squeeze, overwrite=True)