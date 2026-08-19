import os, mrcfile
from dataclasses import dataclass
from easymode.segmentation.augmentations import *
from easymode.segmentation.dataset import open_datasets, close_datasets
import tensorflow as tf
import numpy as np

AUGMENTATIONS_ROT_XZ_YZ = 0.2
AUGMENTATIONS_ROT_XY = 0.2
AUGMENTATIONS_FLIP = 0.5   # set to 0.0 to preserve chirality; a flip mirrors the sample
AUGMENTATIONS_MISSING_WEDGE = 0.0   # NB: remove_wedge re-normalizes per-patch -- fix that before re-enabling under global norm
AUGMENTATIONS_GAUSSIAN = 0.2
AUGMENTATIONS_SCALE = 0.2
AUGMENTATIONS_MIXUP = 0.2
AUGMENTATIONS_CONTRAST = 0.5
AUGMENTATIONS_FLAVOURMIX = 0.67   # fraction of training samples served as a two-flavour mix (else one pure flavour)
AUGMENTATIONS_FLAVOURMIX_FORCE_RAW = 0.1  # when flavour mixing: don't, and serve raw instead.

DATALOADER_WORKERS = 8  # parallel augmentation threads; sharing one process keeps the volume cache shared
IGNORE_BORDER = 8  # voxels of label set to ignore on every face of the crop

_VOLUME_CACHE = {}


@dataclass
class AugConfig:
    """Defaults are the module constants above -- an unset CLI flag is provably the calibrated value."""
    rot_xz_yz: float = AUGMENTATIONS_ROT_XZ_YZ
    rot_xy: float = AUGMENTATIONS_ROT_XY
    flip: float = AUGMENTATIONS_FLIP
    blur: float = AUGMENTATIONS_GAUSSIAN
    scale: float = AUGMENTATIONS_SCALE
    mixup: float = AUGMENTATIONS_MIXUP
    contrast: float = AUGMENTATIONS_CONTRAST
    flavourmix: float = AUGMENTATIONS_FLAVOURMIX
    wedge: float = AUGMENTATIONS_MISSING_WEDGE


def read_volume(path, cache=False, copy=False):
    """copy=True for volumes the caller mutates in place -- otherwise it edits the cached array."""
    if not cache:
        return mrcfile.read(path)
    if path not in _VOLUME_CACHE:
        _VOLUME_CACHE[path] = mrcfile.read(path)
    return _VOLUME_CACHE[path].copy() if copy else _VOLUME_CACHE[path]


class Sample:
    def __init__(self, dataset, idx, cache=False):
        self.dataset = dataset
        self.idx = idx
        self.cache = cache
        self.valid = True
        self.is_positive = False
        self.flavours = {f: dataset.x_path(idx, f) for f in dataset.flavours_for(idx)}
        self.label_path = dataset.y_path(idx)
        self.validity_path = dataset.validity_path(idx)

        label_volume = read_volume(self.label_path, self.cache)
        if label_volume.size == 0:
            print(f'Corrupt/empty label for sample {self.idx} in {self.dataset.name}')
            self.valid = False
        elif not dataset.negative:
            self.is_positive = np.sum(label_volume == 1) > 0

    def _slab(self, img, label, z_depth, validation):
        # 2D annotation on a 3D box: crop a z_depth window and ignore every slice but the annotated one.
        # Done here, before augment(), so the annotated index never has to survive a rotation or a flip.
        sz = img.shape[0]
        d = sz if z_depth is None else min(int(z_depth), sz)
        k = sz // 2
        lo, hi = max(0, k - d + 1 + IGNORE_BORDER), min(k - IGNORE_BORDER, sz - d)
        if hi < lo:  # window too shallow to keep the annotated slice off the ignore border
            o = max(0, min(k - d // 2, sz - d))
        elif validation:
            o = max(lo, min(k - d // 2, hi))
        else:
            o = random.randint(lo, hi)
        img = img[o:o + d]
        slab = np.full(img.shape, 2.0, dtype=np.float32)
        slab[k - o] = label
        return img, slab

    def load(self, validation=False, flavourmix=AUGMENTATIONS_FLAVOURMIX, z_depth=None):
        available_flavours = sorted(self.flavours.keys())
        if validation:
            flavour = self.dataset.annotated_flavour
            img = read_volume(self.flavours[flavour if flavour in self.flavours else available_flavours[0]], self.cache)
        else:
            raw = self.dataset.raw_flavour

            # boxes are already globally normalized at extract; no per-box normalization here
            if len(available_flavours) >= 2 and random.uniform(0.0, 1.0) < flavourmix:
                if random.uniform(0.0, 1.0) < AUGMENTATIONS_FLAVOURMIX_FORCE_RAW and raw in self.flavours:
                    # bypass mixing, serve raw sample instead.
                    img = read_volume(self.flavours[raw], self.cache)
                else:
                    # mix two distinct flavours
                    flavours = random.sample(available_flavours, 2)
                    mixing_factor = random.uniform(0.0, 1.0)
                    img_a = read_volume(self.flavours[flavours[0]], self.cache)
                    img_b = read_volume(self.flavours[flavours[1]], self.cache)
                    img = img_a * mixing_factor + img_b * (1 - mixing_factor)
            else:
                # one pure flavour (matches what inference sees)
                img = read_volume(self.flavours[random.choice(available_flavours)], self.cache)

        label = read_volume(self.label_path, self.cache).astype(np.float32)
        validity = read_volume(self.validity_path, self.cache) if self.validity_path else None
        if validity is not None and validity.shape != label.shape:
            validity = None

        if self.dataset.negative:
            label[label == 1] = 0

        if validity is not None:
            label[validity == 0] = 2

        if label.ndim == 2 and img.ndim == 3:
            img, label = self._slab(img, label, z_depth, validation)
            validity = np.broadcast_to(validity, img.shape) if validity is not None else None
        elif label.shape != img.shape:
            raise ValueError(f'shape mismatch for {self.idx} in {self.dataset.name}: x {img.shape} vs y {label.shape}')

        return img, label, validity


def build_samples(datasets, cache=False):
    samples = []
    for d in datasets:
        print(d.summary())
        for idx in d.ids:
            sample = Sample(d, idx, cache=cache)
            if sample.valid:
                samples.append(sample)
    return samples


class DataLoader:
    def __init__(self, samples, batch_size=8, validation=False, crop_shape=(160, 160, 160), cache=False, aug=None):
        self.batch_size = batch_size
        self.validation = validation
        self.crop_shape = tuple(crop_shape)
        self.cache = cache
        self.aug = aug or AugConfig()
        self.samples = [s for i, s in enumerate(samples) if (i % 20 == 0) == validation]
        self.positive_samples = [s for s in self.samples if s.is_positive]
        self.negative_samples = [s for s in self.samples if not s.is_positive]

        if not self.samples:
            raise ValueError(f'No usable samples for {"validation" if validation else "training"}.')
        if not self.positive_samples:
            print('\033[93m' + f'warning: no positive samples for {"validation" if validation else "training"}' + '\033[0m')
        np.random.shuffle(self.samples)

        print(f'Loaded {len(self.samples)} samples for {"validation" if self.validation else "training"} '
              f'({len(self.positive_samples)} positive, {len(self.negative_samples)} negative)')
        if self.cache:
            n_volumes = sum(len(s.flavours) + 2 for s in self.samples)
            volume_bytes = read_volume(next(iter(self.samples[0].flavours.values()))).nbytes
            print(f'RAM cache on: up to {n_volumes} volumes x {volume_bytes / 1e6:.0f} MB = {n_volumes * volume_bytes / 1e9:.1f} GB once fully populated')

    def load(self, sample, validation=False):
        return sample.load(validation=validation, flavourmix=self.aug.flavourmix, z_depth=self.crop_shape[0])

    def mixup(self, img, label):
        if not self.negative_samples:
            return img, label
        negative_sample = random.choice(self.negative_samples)
        negative_img, _, _ = self.load(negative_sample)
        if negative_img.shape != img.shape:
            return img, label
        mixing_factor = random.uniform(0.0, 0.2)

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
        if random.uniform(0.0, 1.0) < self.aug.flip:
            img, label = flip(img, label)

        # AUGMENTATION 4 - Gaussian filtering
        if random.uniform(0.0, 1.0) < self.aug.blur:
            img, label = filter_gaussian(img, label)

        # AUGMENTATION 5 - rotate by a random angle between -20 and +20 degrees around X or Y axis
        if random.uniform(0.0, 1.0) < self.aug.rot_xz_yz:
            img, label = rotate_continuous_xz_or_yz(img, label)

        # AUGMENTATION 6 - rotate randomly along Z axis
        if random.uniform(0.0, 1.0) < self.aug.rot_xy:
            img, label = rotate_continuous_xy(img, label)

        # AUGMENTATION 7 - missing wedge simulation
        if random.uniform(0.0, 1.0) < self.aug.wedge:
            img, label = remove_wedge(img, label)

        # AUGMENTATION 8 - magnification jitter (90% to 110%)
        if random.uniform(0.0, 1.0) < self.aug.scale:
            img, label = scale(img, label)

        # AUGMENTATION 9 - mixup between the sample and a random negative sample
        if random.uniform(0.0, 1.0) < self.aug.mixup:
            img, label = self.mixup(img, label)

        if random.uniform(0.0, 1.0) < self.aug.contrast:
            img, label = contrast_jitter(img, label)


        return img, label

    def random_crop(self, img, label):
        cz, cy, cx = self.crop_shape
        sz, sy, sx = img.shape[:3]
        z = random.randint(0, max(sz - cz, 0))
        y = random.randint(0, max(sy - cy, 0))
        x = random.randint(0, max(sx - cx, 0))
        return img[z:z+cz, y:y+cy, x:x+cx], label[z:z+cz, y:y+cy, x:x+cx]

    def preprocess(self, img, label, validity):
        img = img.astype(np.float32)

        if validity is not None:
            invalid_voxels_mask = validity == 0
            # 260227: for data sampled at large A/px, much of box has label == 2, where img == 0, throwing off
            # batch normalization. try replacing these voxel values with noise matching valid image statistics
            if invalid_voxels_mask.any() and (label < 2).any():
                img_mu = np.mean(img[label < 2])  # normalization factors for real data only (not for out of bounds ignored areas)
                img_std = np.std(img[label < 2]) # normalization factors for real data only (not for out of bounds ignored areas)
                img[invalid_voxels_mask] = np.random.normal(img_mu, img_std, int(np.sum(invalid_voxels_mask)))

        label = label.astype(np.float32)
        return img, label

    def sample_generator(self, shard=0, n_shards=1):
        e = IGNORE_BORDER
        if self.validation:
            for sample in self.samples:
                img, label, validity = self.load(sample, validation=True)
                img, label = self.preprocess(img, label, validity)
                img, label = self.random_crop(img, label)
                label[:e, :, :] = 2; label[-e:, :, :] = 2
                label[:, :e, :] = 2; label[:, -e:, :] = 2
                label[:, :, :e] = 2; label[:, :, -e:] = 2
                yield np.expand_dims(img, axis=-1), np.expand_dims(label, axis=-1)
        else:
            order = list(self.samples)  # per-shard copy; shuffling self.samples from N threads races
            k = 0
            while True:
                random.shuffle(order)
                for j in range(int(shard), len(order), int(n_shards)):
                    if k % 3 == 0 and self.positive_samples:  # count yields, not indices: a strided j would skew the 1-in-3 ratio
                        sample = random.choice(self.positive_samples)
                    else:
                        sample = order[j]
                    k += 1

                    img, label, validity = self.load(sample)
                    img, label = self.preprocess(img, label, validity)
                    img, label = self.augment(img, label)
                    img, label = self.random_crop(img, label)

                    label[:e, :, :] = 2; label[-e:, :, :] = 2
                    label[:, :e, :] = 2; label[:, -e:, :] = 2
                    label[:, :, :e] = 2; label[:, :, -e:] = 2

                    yield np.expand_dims(img, axis=-1), np.expand_dims(label, axis=-1)

    def as_generator(self, batch_size):
        cz, cy, cx = self.crop_shape
        spec = tf.TensorSpec(shape=(cz, cy, cx, 1), dtype=tf.float32)
        workers = DATALOADER_WORKERS
        if self.validation or workers <= 1:
            dataset = tf.data.Dataset.from_generator(self.sample_generator, output_signature=(spec, spec))
        else:
            # threads, not processes: the workers share one _VOLUME_CACHE, and numpy/scipy release the GIL
            dataset = tf.data.Dataset.range(workers).interleave(
                lambda i: tf.data.Dataset.from_generator(self.sample_generator, output_signature=(spec, spec), args=(i, workers)),
                cycle_length=workers, num_parallel_calls=workers, deterministic=False)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        if self.validation:
            return dataset, None  # finite dataset — Keras runs until exhausted
        n_steps = len(self.samples) // batch_size
        return dataset, n_steps


def _check_geometry(datasets, crop_shape):
    for d in datasets:
        for axis, box, crop in zip('ZYX', d.box_shape, crop_shape):
            if box < crop:
                raise ValueError(f'{d.name}: box is {"x".join(str(s) for s in d.box_shape)} but the crop is '
                                 f'{"x".join(str(s) for s in crop_shape)} ({axis}: {box} < {crop}). Use --size.')
    if crop_shape[0] <= 2 * IGNORE_BORDER:
        raise ValueError(f'Crop depth {crop_shape[0]} leaves no supervised slice inside the {IGNORE_BORDER}-voxel ignore border.')


def train_model(title='', data=(), output=None, batch_size=8, epochs=100, lr_start=1e-3, lr_end=1e-5,
                weights_path=None, bce_weight=0.3, dice_weight=0.7, arch='unet-membrain-groupnorm',
                crop_shape=(160, 160, 160), cache=False, aug=None, apix=None, apix_z=None, normalization=None,
                xla=False):
    import json
    from datetime import datetime, timezone
    from easymode.segmentation.models import get_arch, resolve_arch
    from easymode.segmentation.metrics import SoftMaskedPrecision, SoftMaskedRecall

    arch = resolve_arch(arch)
    m = get_arch(arch)['module']
    create = m.create
    masked_bce_loss = m.masked_bce_loss
    masked_dice_loss = m.masked_dice_loss

    crop_shape = tuple(crop_shape)
    cz, cy, cx = crop_shape
    aug = aug or AugConfig()
    output = output or title

    tf.config.run_functions_eagerly(False)

    datasets = open_datasets(data)
    try:
        _check_geometry(datasets, crop_shape)
        apix = apix or next((d.apix for d in datasets if d.apix), None)
        apix_z = apix_z or next((d.apix_z for d in datasets if d.apix_z), None) or apix
        if apix is None:
            apix, apix_z = 10.0, 10.0
            print('\033[93m' + 'warning: no pixel size in the data or metadata - assuming 10.0 A/px. Use --apix.' + '\033[0m')
        # How the data was normalized is not measurable from the boxes, so it is a claim: --normalization
        # if given, else what the datasets declare. The model is tagged with it and inference obeys the tag.
        norms = {d.normalization for d in datasets}
        if normalization is None:
            if len(norms) > 1:
                print('\033[93m' + f'warning: datasets disagree on input normalization ({norms}); tagging the model as {sorted(norms)[0]}. Use --normalization.' + '\033[0m')
            normalization = sorted(norms)[0]
        elif norms != {normalization}:
            print('\033[93m' + f'warning: --normalization {normalization} overrides the data ({sorted(norms)}); the model is tagged {normalization}' + '\033[0m')

        print(f'\nTraining model {title} (arch: {arch}, crop: {cz}x{cy}x{cx}, {apix:.2f} A/px, norm: {normalization})\n')
        print(f'Loss weights: BCE={bce_weight}, dice={dice_weight}\n')

        def combined_loss(y_true, y_pred):
            return bce_weight * masked_bce_loss(y_true, y_pred) + dice_weight * masked_dice_loss(y_true, y_pred)

        with tf.distribute.MirroredStrategy().scope():
            model = create()
            # Streaming, dataset-level soft precision/recall (threshold-free) for every arch.
            eval_metrics = [SoftMaskedPrecision(), SoftMaskedRecall()]
            model.compile(optimizer=model.optimizer, loss=combined_loss, metrics=eval_metrics, run_eagerly=False)
            # INVARIANT: every variable must be created eagerly, before any tf.function sees the model.
            # A subclassed model built inside a trace names its nested variables off the FuncGraph's uid
            # counter, which restarts per trace and hands two sibling layers the same variable name --
            # fatal only later, when saving to HDF5 (one dataset per weight name): "name already exists".
            model(tf.zeros((1, cz, cy, cx, 1)))
            names = [w.name for w in model.weights]
            if len(names) != len(set(names)):
                dupes = sorted({n for n in names if names.count(n) > 1})
                raise RuntimeError(f'model has duplicate weight names (would fail at first save): {dupes[:4]} '
                                   f'-- the model was built inside a tf.function trace; report this.')
            if weights_path is not None:
                if os.path.isdir(weights_path):
                    # a run's output directory: take its .h5, or an old-style TF checkpoint
                    h5 = sorted(f for f in os.listdir(weights_path) if f.endswith('.h5'))
                    checkpoint = os.path.join(weights_path, h5[0]) if h5 else tf.train.latest_checkpoint(weights_path)
                    if checkpoint is None:
                        raise ValueError(f'No weights (.h5) found in {weights_path}')
                    weights_path = checkpoint
                model.load_weights(weights_path)
            if xla:
                # XLA-compile only the network forward (and thereby its backward): jitting the whole
                # train step would put the metric counters inside the compiled function, where mid-epoch
                # reads race across replicas (impossible P/R > 1 in the progress bar). Wrapped strictly
                # AFTER the eager build + weight load, so no variable is ever created inside the trace.
                model.call = tf.function(model.call, jit_compile=True)
        if hasattr(m, 'lr_schedule'):
            model.optimizer.learning_rate.assign(m.lr_schedule(0, epochs))
        else:
            model.optimizer.learning_rate.assign(lr_start)

        # data loaders — built once and shared, so archives are extracted and labels read only once
        samples = build_samples(datasets, cache=cache)
        training_ds, training_steps = DataLoader(samples, batch_size=batch_size, validation=False, crop_shape=crop_shape, cache=cache, aug=aug).as_generator(batch_size=batch_size)
        validation_ds, validation_steps = DataLoader(samples, batch_size=batch_size, validation=True, crop_shape=crop_shape, cache=cache, aug=aug).as_generator(batch_size=batch_size)

        # callbacks
        os.makedirs(output, exist_ok=True)
        weights_out = os.path.join(output, f'{title}.h5')
        # The sidecar is written up front, so the .h5 the checkpoint callback drops after every
        # improved epoch is immediately usable -- no packaging step, and a killed run still leaves
        # a ready model behind.
        with open(os.path.join(output, f'{title}.json'), 'w', encoding='utf-8') as _f:
            json.dump({'apix': apix,
                       'apix_z': apix_z,
                       'arch': arch,
                       'normalization': normalization,
                       'timestamp': datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}, _f, indent=2)
        cb_checkpoint_train = tf.keras.callbacks.ModelCheckpoint(filepath=weights_out,
                                                                 monitor=f'loss',
                                                                 save_best_only=True,
                                                                 save_weights_only=True,
                                                                 mode='min',
                                                                 verbose=0)

        if hasattr(m, 'lr_schedule'):
            # Arch owns its LR schedule (e.g. nnU-Net PolyLR); CLI lr_start/lr_end ignored.
            def lr_decay(epoch, _):
                return m.lr_schedule(epoch, epochs)
        else:
            def lr_decay(epoch, _):
                return float(lr_start + (lr_end - lr_start) * (epoch / epochs))

        cb_lr = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=0)

        cb_csv = tf.keras.callbacks.CSVLogger(os.path.join(output, 'training_log.csv'), append=False)
        try:
            model.fit(training_ds, steps_per_epoch=training_steps, validation_data=validation_ds, validation_steps=validation_steps, epochs=epochs, validation_freq=1, callbacks=[cb_checkpoint_train, cb_lr, cb_csv])
        finally:
            if os.path.exists(weights_out):
                print(f'\nSegment with this model using: easymode segment --model {weights_out} --data <tomograms>')
    finally:
        close_datasets(datasets)


def preview_samples(data, output='samples', crop_shape=(160, 160, 160), n=30):
    import uuid, shutil
    for sub in ['x_main', 'y']:
        d = os.path.join(output, sub)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    datasets = open_datasets(data)
    try:
        loader = DataLoader(build_samples(datasets), validation=False, crop_shape=crop_shape)
        gen = loader.sample_generator()

        for i in range(n):
            img, label = next(gen)
            img = np.squeeze(img, axis=-1)
            label = np.squeeze(label, axis=-1)

            h = uuid.uuid4().hex[:12]

            with mrcfile.new(os.path.join(output, 'x_main', f'{h}.mrc'), overwrite=True) as mrc:
                mrc.set_data(img)
            with mrcfile.new(os.path.join(output, 'y', f'{h}.mrc'), overwrite=True) as mrc:
                mrc.set_data(label)

            print(f'Saved {i + 1}/{n} - {h} - label values {np.unique(label)}')
    finally:
        close_datasets(datasets)

    print('Done.')
