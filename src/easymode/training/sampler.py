"""
Generic subtomogram sampler used by all (x -> y) supervised denoising/dewedging
trainers in easymode.

A *flavour* is a tomogram source with a known on-disk location. A *mode* pairs
two flavours into a supervised training task (input -> target). To add a new
mode, register one entry in MODES.

Layout assumed by the flavour resolvers:

  datasets/{dataset}/warp_tiltseries/reconstruction/{stem}.mrc         (raw)
  datasets/{dataset}/warp_tiltseries/reconstruction/even/{stem}.mrc    (even)
  datasets/{dataset}/warp_tiltseries/reconstruction/odd/{stem}.mrc     (odd)
  volumes_cryocare/{stem}.mrc                                          (cryocare)
  volumes_ddw/{stem}.mrc                                               (ddw)
  training/isonet2/per_dataset/{dataset}/corrected/_isonet2-n2n_{ARCH}_{stem}.mrc  (iso)

Output layout (one directory per mode, written to ROOT/training/{mode}/):

  training/{mode}/volumes_training/{x,y}/{hash}.mrc
  training/{mode}/volumes_validation/{x,y}/{hash}.mrc

Filenames are md5(mode, dataset, stem, j, k, l)[:12] -- same trick as the
segmentation sampler. Train/val split is decided per-tomogram from a hash of
the tomogram identity, so the assignment is deterministic and worker-local.
This means each ProcessPoolExecutor worker can compute its own output paths
from its inputs alone -- no pre-planning, no shared counters, no index
collisions across processes.

Parallelism: tomograms are processed by a ProcessPoolExecutor. cephfs is
bottlenecked on file-create metadata ops; parallelising the writes is what
makes this finish in minutes rather than hours.
"""
import glob, hashlib, os, random, shutil, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import mrcfile

DEBUG = False
ROOT = "/cephfs/mlast/compu_projects/easymode"
if os.name == "nt":
    DEBUG = True
    ROOT = "C:/Users/Mart Last/Desktop/easymode"

# arch tag baked into the IsoNet2 corrected/ filenames (must match training/isonet2/train.py ARCH)
ISONET2_ARCH = "unet-medium"


# ----- flavour resolvers -------------------------------------------------------

def _path_raw(dataset, stem):
    return f"{ROOT}/datasets/{dataset}/warp_tiltseries/reconstruction/{stem}.mrc"


def _path_even(dataset, stem):
    return f"{ROOT}/datasets/{dataset}/warp_tiltseries/reconstruction/even/{stem}.mrc"


def _path_odd(dataset, stem):
    return f"{ROOT}/datasets/{dataset}/warp_tiltseries/reconstruction/odd/{stem}.mrc"


def _path_cryocare(dataset, stem):
    return f"{ROOT}/volumes_cryocare/{stem}.mrc"


def _path_ddw(dataset, stem):
    return f"{ROOT}/volumes_ddw/{stem}.mrc"


def _path_iso(dataset, stem):
    return (f"{ROOT}/training/isonet2/per_dataset/{dataset}/corrected/"
            f"_isonet2-n2n_{ISONET2_ARCH}_{stem}.mrc")


FLAVOURS = {
    "raw":      _path_raw,
    "even":     _path_even,
    "odd":      _path_odd,
    "cryocare": _path_cryocare,
    "ddw":      _path_ddw,
    "iso":      _path_iso,
}


# ----- mode registry: (x flavour, y flavour) -----------------------------------

MODES = {
    # noise2noise on even/odd half-map pairs -- the original n2n splits trainer.
    "n2n": ("even", "odd"),

    # distillation of the per-dataset DDW2 teachers into one fast general student:
    # raw full tomogram -> wedge-filled, denoised target produced by `ddw refine-tomogram`.
    "ddw": ("raw", "ddw"),

    # distillation of the per-dataset IsoNet2 teachers into one general student:
    # raw full tomogram -> IsoNet2-corrected target (training/isonet2/per_dataset/*/corrected/).
    # No per-tomogram curation here -- every tomogram with both flavours is sampled.
    "iso": ("raw", "iso"),
}


def discover_stems(dataset):
    """Enumerate tomogram stems for a dataset by listing the even/ folder (every
    dataset we train on has even/odd halves; raw and other flavours derive
    their stems from these names)."""
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(_path_even(dataset, "*")))


def _box_id(mode, dataset, stem, j, k, l):
    """Deterministic 12-hex-char filename for one box."""
    key = f"{mode}|{dataset}|{stem}|{j}|{k}|{l}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _split_for_tomo(dataset, stem, val_every, seed):
    """Per-tomogram train/val assignment. Hash-based so workers agree without
    coordination; bucket 0 of val_every -> validation, otherwise training."""
    h = int(hashlib.md5(f"{seed}|{dataset}|{stem}".encode()).hexdigest(), 16)
    return "validation" if h % val_every == 0 else "training"


# ----- coordinate sampling -----------------------------------------------------

def sample_coordinates(shape, box_size, n_samples, rng):
    """XY grid + central-Z slab placement. `rng` is a random.Random instance so
    each worker samples deterministically from its own seed."""
    n_y = shape[1] // box_size
    n_x = shape[2] // box_size
    coords = [(0, j * box_size, i * box_size) for j in range(n_y) for i in range(n_x)]
    rng.shuffle(coords)

    while len(coords) < n_samples:
        coords.append((0, rng.randint(0, shape[1] - box_size),
                          rng.randint(0, shape[2] - box_size)))

    if shape[0] < box_size * 2:
        coords = [(rng.randint(0, shape[0] - box_size), y, x) for (_, y, x) in coords]
    else:
        coords = [(shape[0] // 2 + rng.randint(-box_size, 0), y, x) for (_, y, x) in coords]

    return coords[:n_samples]


# ----- worker (runs in a ProcessPoolExecutor) ----------------------------------

def _worker(task):
    """Process one tomogram independently: decide split, sample coords, extract
    boxes, write them under hash-derived filenames. Returns a (status, records)
    pair so the parent can aggregate into the manifest .star file."""
    (dataset, stem, n_samples, x_flavour, y_flavour, box_size, mode,
     val_every, seed) = task

    x_path = FLAVOURS[x_flavour](dataset, stem)
    y_path = FLAVOURS[y_flavour](dataset, stem)
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return (dataset, stem, "-", 0, "missing flavour", [])

    bs = box_size
    split = _split_for_tomo(dataset, stem, val_every, seed)
    out_x = f"{ROOT}/training/{mode}/volumes_{split}/x"
    out_y = f"{ROOT}/training/{mode}/volumes_{split}/y"
    # per-tomogram RNG seed so coord sampling is deterministic across reruns
    tomo_seed = int(hashlib.md5(f"{seed}|coords|{dataset}|{stem}".encode()).hexdigest(), 16)
    rng = random.Random(tomo_seed & 0xFFFFFFFF)

    records = []
    # mmap: each box needs only its 96**3 slice, not the whole GB volume.
    with mrcfile.mmap(x_path, mode='r', permissive=True) as mx, \
         mrcfile.mmap(y_path, mode='r', permissive=True) as my:
        sx, sy = mx.data.shape, my.data.shape
        if sx != sy:
            return (dataset, stem, split, 0, f"shape mismatch {sx} vs {sy}", [])
        if min(sx) <= bs:
            return (dataset, stem, split, 0, f"volume too small {sx}", [])

        coords = sample_coordinates(sx, bs, n_samples, rng)
        # Propagate the source pixel size to the boxes: mrcfile.new() defaults voxel_size to 0,
        # so without this the boxes carry 0 A/px. Reconstructions are 10 A/px -- fall back to that
        # if the source header is unset.
        apix = float(mx.voxel_size.x) or 10.0
        n_written = 0
        for (j, yy, xx) in coords:
            box_x = np.asarray(mx.data[j:j+bs, yy:yy+bs, xx:xx+bs]).astype(np.float32)
            box_y = np.asarray(my.data[j:j+bs, yy:yy+bs, xx:xx+bs]).astype(np.float32)
            bid = _box_id(mode, dataset, stem, j, yy, xx)
            with mrcfile.new(f"{out_x}/{bid}.mrc", overwrite=True) as f:
                f.set_data(box_x); f.voxel_size = apix
            with mrcfile.new(f"{out_y}/{bid}.mrc", overwrite=True) as f:
                f.set_data(box_y); f.voxel_size = apix
            records.append((bid, dataset, stem, split, int(j), int(yy), int(xx)))
            n_written += 1
    return (dataset, stem, split, n_written, None, records)


# ----- the sampler ------------------------------------------------------------

class Sampler:
    """Walks `datasets/{d}/` in alphabetical order, dispatches a
    ProcessPoolExecutor over tomograms. Each worker decides its own split and
    output filenames from hashes -- no shared state, no coordination."""

    def __init__(self, mode, samples_per_dataset=500, box_size=96,
                 val_every=10, seed=42, num_workers=None,
                 exclude=()):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; known: {sorted(MODES)}")
        self.mode = mode
        self.x_flavour, self.y_flavour = MODES[mode]
        self.samples_per_dataset = samples_per_dataset
        self.box_size = box_size
        self.val_every = val_every
        self.seed = seed
        self.num_workers = num_workers if num_workers else min(32, os.cpu_count() or 8)
        # Datasets the user wants skipped entirely (bad teacher quality).
        # Matched as PREFIXES: '013' matches '013_DIAT'.
        self.exclude = tuple(s.strip() for s in exclude if s.strip())

    def _matches_any(self, dataset, prefixes):
        return any(dataset == p or dataset.startswith(p + "_") or dataset.startswith(p) for p in prefixes)

    def _should_exclude(self, dataset):
        return self._matches_any(dataset, self.exclude)

    def _output_root(self):
        return f"{ROOT}/training/{self.mode}"

    def _enumerate_datasets(self):
        for ds_dir in sorted(glob.glob(f"{ROOT}/datasets/*/")):
            dataset = os.path.basename(ds_dir.rstrip("/"))
            if self._should_exclude(dataset):
                continue
            stems = discover_stems(dataset)
            usable = [s for s in stems
                      if os.path.exists(FLAVOURS[self.x_flavour](dataset, s))
                      and os.path.exists(FLAVOURS[self.y_flavour](dataset, s))]
            if usable:
                yield dataset, usable

    def _build_tasks(self, inventory):
        """One task per tomogram with its sample budget. No pre-planning of
        splits or indices -- the worker decides those locally from hashes."""
        tasks = []
        for dataset, stems in inventory:
            base, rem = divmod(self.samples_per_dataset, len(stems))
            # alphabetised so the budget distribution is deterministic across reruns
            for j, stem in enumerate(sorted(stems)):
                n = base + (1 if j < rem else 0)
                if n == 0:
                    continue
                tasks.append((dataset, stem, n,
                              self.x_flavour, self.y_flavour, self.box_size,
                              self.mode, self.val_every, self.seed))
        return tasks

    def generate(self):
        out = self._output_root()
        for split in ("training", "validation"):
            for side in ("x", "y"):
                d = f"{out}/volumes_{split}/{side}"
                shutil.rmtree(d, ignore_errors=True)
                os.makedirs(d, exist_ok=True)

        print(f"[sampler] mode={self.mode}  x={self.x_flavour}  y={self.y_flavour}")
        print(f"[sampler] writing to {out}/volumes_{{training,validation}}/{{x,y}}/")
        print(f"[sampler] target {self.samples_per_dataset} boxes/dataset, box={self.box_size}")
        if self.exclude:
            print(f"[sampler] excluding datasets matching: {', '.join(self.exclude)}")

        inventory = list(self._enumerate_datasets())
        if not inventory:
            print("[sampler] no datasets have both flavours -- nothing to do.")
            return
        tasks = self._build_tasks(inventory)
        n_train = sum(1 for d, s in [(t[0], t[1]) for t in tasks]
                      if _split_for_tomo(d, s, self.val_every, self.seed) == "training")
        n_val   = len(tasks) - n_train
        print(f"[sampler] {len(inventory)} datasets eligible, {len(tasks)} tomograms "
              f"({n_train} train / {n_val} val)")
        print(f"[sampler] dispatching across {self.num_workers} workers\n")

        all_records = []
        done = 0
        total = len(tasks)
        t0 = time.time()
        step = max(1, total // 50)                       # ~50 flushed progress lines over the run
        with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
            for ds, stem, split, n, err, recs in ex.map(_worker, tasks):
                done += 1
                all_records.extend(recs)
                if err:                                  # always surface skips
                    print(f"  SKIP [{done}/{total}] {ds}/{stem}: {err}", flush=True)
                elif done % step == 0 or done == total:
                    el = time.time() - t0
                    rate = done / max(el, 1e-9)
                    eta = (total - done) / max(rate, 1e-9)
                    print(f"  [{done}/{total}] {done*100//total}%  {len(all_records)} boxes  "
                          f"{rate:.1f} tomo/s  {el:.0f}s elapsed  ETA {eta:.0f}s", flush=True)

        # Count boxes actually on disk -- single source of truth, handles skips.
        n_train_boxes = len(glob.glob(f"{out}/volumes_training/x/*.mrc"))
        n_val_boxes   = len(glob.glob(f"{out}/volumes_validation/x/*.mrc"))
        print(f"\n[sampler] done: {n_train_boxes} train + {n_val_boxes} val boxes  -> {out}/")

        # Manifest: one row per saved (x, y) pair. Lets a later viewer find the
        # parent tomogram + dataset + box coordinates given just the hash.
        self._write_manifest(out, all_records)

    def _write_manifest(self, out, records):
        import starfile, pandas as pd
        if not records:
            print("[sampler] no records to manifest; skipping star.")
            return
        df = pd.DataFrame(records, columns=[
            "hash", "dataset", "tomogram", "split",
            "z", "y", "x",
        ])
        manifest = f"{out}/manifest.star"
        starfile.write(df, manifest, overwrite=True)
        print(f"[sampler] manifest -> {manifest}  ({len(df)} rows)")


def generate(mode, samples_per_dataset=500, box_size=96, num_workers=None,
             exclude=()):
    """Convenience entry point."""
    Sampler(mode=mode, samples_per_dataset=samples_per_dataset, box_size=box_size,
            num_workers=num_workers, exclude=exclude).generate()
