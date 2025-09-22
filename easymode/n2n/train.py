import glob, os, mrcfile, shutil
from easymode.segmentation.augmentations import *
import tensorflow as tf

ROOT = 'C:/Users/mart_/Desktop/easymode'

class N2NDatasetGenerator:
    def __init__(self, mode='splits', samples_per_tomogram=10, box_size=96):
        self.mode = mode
        self.samples_per_tomogram = samples_per_tomogram
        self.box_size = box_size
        self.vols_one = []
        self.vols_two = []
        self.box_counter = {'training': 0, 'validation': 0}
        self.tomo_counter = 0

    @staticmethod
    def get_sample_coordinates(shape, box_size, n_samples):
        # first, distribute as many as possible evenly
        n_y = shape[1] // box_size
        n_x = shape[2] // box_size
        coordinates = []
        for j in range(n_y):
            for i in range(n_x):
                coordinates.append((0, j * box_size, i * box_size))
        np.random.shuffle(coordinates)

        # if not enough samples (because the XY plane doesn't fit enough), randomly position some more:
        if len(coordinates) < n_samples:
            while len(coordinates) < n_samples:
                coordinates.append((0, random.randint(0, shape[1] - box_size), random.randint(0, shape[2] - box_size)))

        # now assign the z coordinate. Two cases, if shape[0] < 2 * box_size, just place them randomly; part of it will overlap with the non void content. Else, place in the center 2 * BOX_SIZE slab.
        if shape[0] < box_size * 2:
            for i in range(len(coordinates)):
                coordinates[i] = (random.randint(0, shape[0] - box_size), coordinates[i][1], coordinates[i][2])
        else:
            for i in range(len(coordinates)):
                coordinates[i] = (shape[0] // 2 + random.randint(-box_size, 0), coordinates[i][1], coordinates[i][2])

        return coordinates[:n_samples]

    def sample_tomogram_pair(self, vol_a_path, vol_b_path):
        vol_a = mrcfile.read(vol_a_path)
        vol_b = mrcfile.read(vol_b_path)
        coordinates = self.get_sample_coordinates(vol_a.data.shape, self.box_size, self.samples_per_tomogram)
        self.tomo_counter += 1
        split = 'validation' if self.tomo_counter % 10 == 0 else 'training'

        for (j, k, l) in coordinates:
            box_a = vol_a[j:j+self.box_size, k:k+self.box_size, l:l+self.box_size]
            box_b = vol_b[j:j + self.box_size, k:k + self.box_size, l:l + self.box_size]
            if self.mode == 'splits':
                with mrcfile.new(f'{ROOT}/training/n2n/{self.mode}/volumes_{split}/x/{self.box_counter[split]}.mrc', overwrite=True) as m:
                    m.set_data(box_a.astype(np.float32))
                with mrcfile.new(f'{ROOT}/training/n2n/{self.mode}/volumes_{split}/y/{self.box_counter[split]}.mrc', overwrite=True) as m:
                    m.set_data(box_b.astype(np.float32))
            elif self.mode == 'direct':
                box = (box_a + box_b) / 2.0
                with mrcfile.new(f'{ROOT}/training/n2n/{self.mode}/volumes_{split}/x/{self.box_counter[split]}.mrc', overwrite=True) as m:
                    m.set_data(box.astype(np.float32))
                with mrcfile.new(f'{ROOT}/training/n2n/{self.mode}/volumes_{split}/x/even/{self.box_counter[split]}.mrc', overwrite=True) as m:
                    m.set_data(box_a.astype(np.float32))
                with mrcfile.new(f'{ROOT}/training/n2n/{self.mode}/volumes_{split}/x/odd/{self.box_counter[split]}.mrc', overwrite=True) as m:
                    m.set_data(box_b.astype(np.float32))

            self.box_counter[split] += 1


    def generate_direct_mode_outputs(self):
        print(f'Applying easymode n2n in split mode to the even/odd pairs to generate direct mode training data.')
        gpus = ','.join(str(i) for i in range(len(tf.config.list_physical_devices('GPU'))))
        for split in ['training', 'validation']:
            input_dir = f'{ROOT}/training/n2n/{self.mode}/volumes_{split}/x/'
            output_dir = f'{ROOT}/training/n2n/{self.mode}/volumes_{split}/y/'

            print(f'easymode denoise --data {input_dir} --output {output_dir} --tta 4 --mode splits --overwrite --gpu {gpus}')
            os.system(f'easymode denoise --data {input_dir} --output {output_dir} --tta 4 --mode splits --overwrite --gpu {gpus}')

            shutil.rmtree(os.path.join(input_dir, 'even'))
            shutil.rmtree(os.path.join(input_dir, 'odd'))

    def generate(self):
        print(
            f'Preparing to generate training data for denoiser mode {self.mode} with {self.samples_per_tomogram} samples per tomogram.\n')

        if self.mode == 'splits':
            base_path = f'{ROOT}/training/n2n/splits'
            shutil.rmtree(f'{base_path}/volumes_training', ignore_errors=True)
            shutil.rmtree(f'{base_path}/volumes_validation', ignore_errors=True)

            os.makedirs(f'{base_path}/volumes_training/x', exist_ok=True)
            os.makedirs(f'{base_path}/volumes_training/y', exist_ok=True)
            os.makedirs(f'{base_path}/volumes_validation/x', exist_ok=True)
            os.makedirs(f'{base_path}/volumes_validation/y', exist_ok=True)
        elif self.mode == 'direct':
            base_path = f'{ROOT}/training/n2n/direct'
            shutil.rmtree(f'{base_path}/volumes_training', ignore_errors=True)
            shutil.rmtree(f'{base_path}/volumes_validation', ignore_errors=True)

            os.makedirs(f'{base_path}/volumes_training/x', exist_ok=True)
            os.makedirs(f'{base_path}/volumes_training/x/even', exist_ok=True)
            os.makedirs(f'{base_path}/volumes_training/x/odd', exist_ok=True)
            os.makedirs(f'{base_path}/volumes_training/y', exist_ok=True)
            os.makedirs(f'{base_path}/volumes_validation/x', exist_ok=True)
            os.makedirs(f'{base_path}/volumes_validation/x/even', exist_ok=True)
            os.makedirs(f'{base_path}/volumes_validation/x/odd', exist_ok=True)
            os.makedirs(f'{base_path}/volumes_validation/y', exist_ok=True)

        self.load_splits()

        for j, (vol_a, vol_b) in enumerate(zip(self.vols_one, self.vols_two)):
            print(f'{j + 1}/{len(self.vols_one)}: {vol_a}')
            self.sample_tomogram_pair(vol_a, vol_b)

        if self.mode == 'direct':
            self.generate_direct_mode_outputs()

    def load_splits(self):
        datasets = [os.path.basename(os.path.dirname(f)) for f in glob.glob(f'{ROOT}/datasets/*/')]

        print(f'Found {len(datasets)} datasets to sample.')

        np.random.shuffle(datasets)
        self.vols_one = list()
        self.vols_two = list()

        for d in datasets:
            tomograms = [os.path.basename(f) for f in glob.glob(f'{ROOT}/datasets/{d}/warp_tiltseries/reconstruction/even/*.mrc')]
            np.random.shuffle(tomograms)
            for t in tomograms:
                path_evn = f'{ROOT}/datasets/{d}/warp_tiltseries/reconstruction/even/{t}'
                path_odd = f'{ROOT}/datasets/{d}/warp_tiltseries/reconstruction/odd/{t}'
                if os.path.exists(path_evn) and os.path.exists(path_odd):
                    self.vols_one.append(path_evn)
                    self.vols_two.append(path_odd)



class N2NDataloader:
    def __init__(self, mode='splits', batch_size=32, box_size=96, validation=False):
        self.mode = mode
        self.batch_size = batch_size
        self.box_size = box_size
        self.validation = validation
        self.indices = list()
        self.parse_indices()

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

    def parse_indices(self):
        sample_directory = f'{ROOT}/training/n2n/{self.mode}/volumes_{"validation" if self.validation else "training"}/x/'
        sample_names = sorted([os.path.basename(f) for f in glob.glob(f'{sample_directory}/*.mrc')])
        self.indices = list(range(len(sample_names)))

    def get_sample(self, index):
        train_x = mrcfile.read(f'{ROOT}/training/n2n/{self.mode}/volumes_{"validation" if self.validation else "training"}/x/{index}.mrc').data
        train_y = mrcfile.read(f'{ROOT}/training/n2n/{self.mode}/volumes_{"validation" if self.validation else "training"}/y/{index}.mrc').data

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)

        return train_x, train_y

    def sample_generator(self):
        while True:
            np.random.shuffle(self.indices)
            for j in self.indices:
                train_x, train_y = self.get_sample(j)
                train_x, train_y = self.augment(train_x, train_y, self.validation)
                train_x, train_y = self.preprocess(train_x, train_y)

                if not self.validation and self.mode == 'splits' and np.random.rand() < 0.5:  # when training from odd/even splits, train_x and train_y can be swapped.
                    yield train_y, train_x
                else:
                    yield train_x, train_y

    def as_generator(self, batch_size, num_epochs=None):
        dataset = tf.data.Dataset.from_generator(
            self.sample_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.box_size, self.box_size, self.box_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(self.box_size, self.box_size, self.box_size, 1), dtype=tf.float32)
            )
        )

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        options.threading.private_threadpool_size = 32
        dataset = dataset.with_options(options)

        if num_epochs:
            dataset = dataset.repeat(num_epochs)

        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        n_steps = len(self.indices) // batch_size
        return dataset, n_steps


def train_n2n(mode='splits', batch_size=32, box_size=96, epochs=100, lr_start=1e-3, lr_end=1e-5):
    from easymode.n2n.model import create

    tf.config.run_functions_eagerly(False)

    print(f'\nTraining n2n model for mode: {mode}\n')

    with tf.distribute.MirroredStrategy().scope():
        model = create()
    model.optimizer.learning_rate.assign(lr_start)

    # data loaders
    training_ds, training_steps = N2NDataloader(mode=mode, batch_size=batch_size, box_size=box_size, validation=False).as_generator(batch_size=batch_size)
    validation_ds, validation_steps = N2NDataloader(mode=mode, batch_size=batch_size, box_size=box_size, validation=True).as_generator(batch_size=batch_size)

    # callbacks
    os.makedirs(f'{ROOT}/training/n2n/{mode}/checkpoints/', exist_ok=True)
    # cb_checkpoint_val = tf.keras.callbacks.ModelCheckpoint(filepath=f'{ROOT}/training/n2n/{mode}/checkpoints/' + "validation_loss",
    #                                                        monitor=f'val_loss',
    #                                                        save_best_only=True,
    #                                                        save_weights_only=True,
    #                                                        mode='min',
    #                                                        verbose=1)
    cb_checkpoint_train = tf.keras.callbacks.ModelCheckpoint(filepath=f'{ROOT}/training/n2n/{mode}/checkpoints/' + "training_loss",
                                                             monitor=f'loss',
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode='min',
                                                             verbose=1)

    def lr_decay(epoch, _):
        return float(lr_start + (lr_end - lr_start) * ((epoch - 2) / epochs))

    cb_lr = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=1)

    cb_csv = tf.keras.callbacks.CSVLogger(f'{ROOT}/training/n2n/{mode}/checkpoints/training_log.csv', append=True)

    model.fit(training_ds, steps_per_epoch=training_steps, validation_data=validation_ds, validation_steps=validation_steps, epochs=epochs, validation_freq=1, callbacks=[cb_checkpoint_val, cb_checkpoint_train, cb_lr, cb_csv])

