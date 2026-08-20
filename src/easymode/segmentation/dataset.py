"""Training data format. A dataset is a directory or an (uncompressed) tar archive -- .tar
or the Ais .scnt -- laid out as:

    x_<flavour>/<id>.mrc    one or more input flavours; any directory named x_* is one
    y/<id>.mrc              labels: 0 = negative, 1 = positive, 2 = ignore
    validity/<id>.mrc       optional uint8 mask, 0 = out of bounds -> folded in as label 2
    metadata.json           optional; no field is required

Files are joined by filename. Sample ids come from y/, so a flavour that covers only part of
the set is fine. x and y must have the same shape, except that a 2D y with a 3D x is expanded
into a slab label at load time (see train.Sample._slab)."""
import atexit, glob, json, os, shutil, tarfile, tempfile
import mrcfile
from easymode.segmentation.normalization import NORM_GLOBAL_MAD

INPUT_PREFIX = 'x_'
LABEL_DIR = 'y'
VALIDITY_DIR = 'validity'
DEFAULT_ANNOTATED_FLAVOUR = 'x_main'
RAW_FLAVOUR = 'x_raw'


class TrainingDataset:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(os.path.normpath(path)).split('.')[0]
        self._tmp = None
        self.root = self._resolve_root(path)

        self.meta = {}
        meta_file = os.path.join(self.root, 'metadata.json')
        if os.path.exists(meta_file):
            with open(meta_file, encoding='utf-8') as f:
                self.meta = json.load(f)

        self.flavours = sorted(d for d in os.listdir(self.root)
                               if d.startswith(INPUT_PREFIX) and os.path.isdir(os.path.join(self.root, d)))
        if not os.path.isdir(os.path.join(self.root, LABEL_DIR)):
            raise ValueError(f'{path}: no {LABEL_DIR}/ directory - see the training data format in the docs.')
        if not self.flavours:
            raise ValueError(f'{path}: no {INPUT_PREFIX}* input directories - see the training data format in the docs.')

        self.ids = sorted(os.path.splitext(n)[0] for n in os.listdir(os.path.join(self.root, LABEL_DIR))
                          if n.endswith('.mrc'))
        if not self.ids:
            raise ValueError(f'{path}: {LABEL_DIR}/ contains no .mrc files.')

        self._coverage = {i: [] for i in self.ids}
        for f in self.flavours:
            present = {os.path.splitext(n)[0] for n in os.listdir(os.path.join(self.root, f)) if n.endswith('.mrc')}
            for i in self.ids:
                if i in present:
                    self._coverage[i].append(f)
        orphans = [i for i in self.ids if not self._coverage[i]]
        if orphans:
            print(f'{self.name}: dropping {len(orphans)} sample(s) with no input volume (e.g. {orphans[0]})')
            self.ids = [i for i in self.ids if self._coverage[i]]
        if not self.ids:
            raise ValueError(f'{path}: no sample has an input volume in any {INPUT_PREFIX}* directory.')

        annotated = self.meta.get('annotated_flavour')
        if annotated not in self.flavours:
            annotated = DEFAULT_ANNOTATED_FLAVOUR if DEFAULT_ANNOTATED_FLAVOUR in self.flavours else self.flavours[0]
        self.annotated_flavour = annotated
        self.raw_flavour = RAW_FLAVOUR if RAW_FLAVOUR in self.flavours else None
        self.has_validity = os.path.isdir(os.path.join(self.root, VALIDITY_DIR))
        self.normalization = self.meta.get('normalization', NORM_GLOBAL_MAD)

        self._read_geometry()

    def _resolve_root(self, path):
        if os.path.isdir(path):
            return path
        if not os.path.exists(path):
            raise ValueError(f'{path}: no such file or directory.')
        if not tarfile.is_tarfile(path):
            if path.endswith('.scnt'):
                raise ValueError(f'{path}: legacy (TIFF) .scnt training sets are not supported - re-extract with a current Ais.')
            raise ValueError(f'{path}: not a directory and not a tar archive.')

        # extract beside the archive: same filesystem as the data, so the space is where the user put it
        size = os.path.getsize(path)
        try:
            self._tmp = tempfile.mkdtemp(prefix='.easymode_train_', dir=os.path.dirname(os.path.abspath(path)))
        except OSError:
            self._tmp = tempfile.mkdtemp(prefix='easymode_train_')
        atexit.register(self.close)
        free = shutil.disk_usage(self._tmp).free
        if free < 1.1 * size:
            raise ValueError(f'{path}: need ~{size / 1e9:.1f} GB to extract but only {free / 1e9:.1f} GB is free at '
                             f'{self._tmp}. Extract the archive yourself and pass the directory instead.')
        print(f'Extracting {path} ({size / 1e9:.1f} GB) to {self._tmp}')
        with tarfile.open(path, 'r') as tar:
            if hasattr(tarfile, 'data_filter'):
                tar.extractall(self._tmp, filter='data')
            else:
                tar.extractall(self._tmp)

        root = self._tmp
        entries = os.listdir(root)
        if len(entries) == 1 and os.path.isdir(os.path.join(root, entries[0])):
            root = os.path.join(root, entries[0])  # tar of a directory rather than of its contents
        return root

    def _read_geometry(self):
        idx = self.ids[0]
        with mrcfile.open(self.x_path(idx, self._coverage[idx][0]), header_only=True, permissive=True) as m:
            self.box_shape = (int(m.header.nz), int(m.header.ny), int(m.header.nx))
            apix = float(m.voxel_size.x)
            apix_z = float(m.voxel_size.z)
        self.apix = self.meta.get('apix') or (apix if apix > 0 else None)
        self.apix_z = self.meta.get('apix_z') or (apix_z if apix_z > 0 else None)

    def x_path(self, idx, flavour):
        return os.path.join(self.root, flavour, f'{idx}.mrc')

    def y_path(self, idx):
        return os.path.join(self.root, LABEL_DIR, f'{idx}.mrc')

    def validity_path(self, idx):
        if not self.has_validity:
            return None
        p = os.path.join(self.root, VALIDITY_DIR, f'{idx}.mrc')
        return p if os.path.exists(p) else None

    def flavours_for(self, idx):
        return self._coverage[idx]

    def summary(self):
        cov = ' '.join(f'{f}={sum(1 for i in self.ids if f in self._coverage[i])}' for f in self.flavours)
        apix = f'{self.apix:.2f} A/px' if self.apix else 'unknown A/px'
        return (f'{self.name}: {len(self.ids)} samples, {"x".join(str(s) for s in self.box_shape)}, {apix}'
                f'\n  flavours: {cov} (annotated: {self.annotated_flavour})')

    def close(self):
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None


def open_datasets(patterns):
    """patterns: dirs, archives, or globs. A directory that holds several datasets expands to its children."""
    paths = []
    for p in ([patterns] if isinstance(patterns, str) else list(patterns)):
        matches = sorted(glob.glob(p)) if glob.has_magic(p) else [p]
        if not matches:
            raise ValueError(f'{p}: no match.')
        for m in matches:
            if os.path.isdir(m) and not os.path.isdir(os.path.join(m, LABEL_DIR)):
                children = sorted(c for c in glob.glob(os.path.join(m, '*'))
                                  if os.path.isdir(os.path.join(c, LABEL_DIR)))
                if children:
                    paths.extend(children)
                    continue
            paths.append(m)

    datasets = []
    try:
        for p in paths:
            datasets.append(TrainingDataset(p))
    except BaseException:
        close_datasets(datasets)
        raise
    return datasets


def close_datasets(datasets):
    for d in datasets:
        d.close()
