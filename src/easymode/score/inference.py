import os, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
import mrcfile
import starfile
from scipy.ndimage import zoom
from easymode.core.distribution import get_model, load_model
from easymode.segmentation.inference import collect_tomograms

PATCH, OVERLAP, MODEL_APIX = 96, 0.1, 10.0
BATCH_SIZE = 16
COLUMNS = ('alignment_score', 'alignment_spread')
POM_SUMMARY = os.path.join('pom', 'summary.star')
DEFAULT_OUT = 'alignment_scores.star'


def _grid(n, patch, overlap):
    if n <= patch:
        return [0]
    stride = max(1, int(patch * (1 - overlap)))
    k = math.ceil((n - patch) / stride) + 1
    return [int(round(v)) for v in np.linspace(0, n - patch, k)]


def load_volume(path, input_apix=None):
    with mrcfile.open(path, permissive=True) as m:
        vol = np.asarray(m.data, dtype=np.float32)
        apix = float(input_apix or m.voxel_size.x)
    if apix <= 1.0:
        apix = MODEL_APIX
    if abs(apix - MODEL_APIX) > 0.05:
        vol = zoom(vol, apix / MODEL_APIX, order=1)
    return vol, apix


def iter_patches(vol, patch=PATCH, overlap=OVERLAP):
    z0 = max(0, (vol.shape[0] - patch) // 2)
    vol = vol[z0:z0 + patch]
    pad = [(0, max(0, patch - s)) for s in vol.shape]
    if any(p[1] for p in pad):
        vol = np.pad(vol, pad, mode='reflect')
    for y in _grid(vol.shape[1], patch, overlap):
        for x in _grid(vol.shape[2], patch, overlap):
            p = vol[:, y:y + patch, x:x + patch]
            p = (p - p.mean()) / max(float(p.std()), 1e-6)
            yield (z0 + patch // 2, y + patch // 2, x + patch // 2), p


def score_volume(model, vol, batch_size=BATCH_SIZE):
    centers, scores, logp, batch = [], [], [], []

    def flush():
        s, lp = model(np.stack(batch)[..., None], training=False)
        scores.append(np.asarray(s)[:, 0])
        logp.append(np.asarray(lp)[:, 0])
        batch.clear()

    for c, p in iter_patches(vol):
        centers.append(c)
        batch.append(p)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()

    scores, logp = np.concatenate(scores), np.concatenate(logp)
    prec = np.exp(np.clip(logp, -20, 20))
    w = prec / prec.sum()
    mean = float((w * scores).sum())
    spread = float(np.sqrt((w * (scores - mean) ** 2).sum()))
    eff = float(prec.sum() ** 2 / (prec ** 2).sum())
    return dict(score=mean, spread=spread, n_patches=len(scores), eff_patches=eff,
                centers=np.array(centers), scores=scores, precisions=prec)


def score_tomogram(model, path, input_apix=None, batch_size=BATCH_SIZE):
    vol, apix = load_volume(path, input_apix)
    r = score_volume(model, vol, batch_size)
    r['apix'] = apix
    return r


def _read_star(path):
    if not os.path.exists(path):
        return pd.DataFrame(index=pd.Index([], name='tomogram'))
    df = starfile.read(path, parse_as_string=['tomogram']).set_index('tomogram')
    df.index = df.index.astype(str)
    return df


def _write_star(df, path):
    df = df.reset_index()
    df['tomogram'] = df['tomogram'].astype(str)
    starfile.write(df, path)


def dispatch(data, model=None, out=None, overwrite=False, gpu=None):
    gpus = tf.config.list_physical_devices('GPU')
    if gpu is not None and gpus:
        gpus = [gpus[int(gpu)]]
        tf.config.set_visible_devices(gpus, 'GPU')
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, True)

    out = out or (POM_SUMMARY if os.path.exists(POM_SUMMARY) else DEFAULT_OUT)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    tomograms = collect_tomograms(data if isinstance(data, (list, tuple)) else [data])
    df = _read_star(out)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    done = set() if overwrite else set(df.index[df['alignment_score'].notna()])
    todo = [t for t in tomograms if os.path.splitext(os.path.basename(t))[0] not in done]
    print(f'Found {len(tomograms)} tomograms, {len(todo)} to score. Writing to {out}\n')
    if not todo:
        return

    if model is None:
        model, _ = get_model('misalignment')
        if model is None:
            return
    print(f'Using model: {model}\n')
    net = load_model(model)

    for j, path in enumerate(todo, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            r = score_tomogram(net, path)
        except Exception as e:
            print(f'{j}/{len(todo)} - {name} - ERROR: {e}')
            continue
        df.loc[name, list(COLUMNS)] = [round(r['score'], 4), round(r['spread'], 4)]
        _write_star(df, out)
        print(f"{j}/{len(todo)} - {name} - score {r['score']:.3f} (spread {r['spread']:.3f}, {r['n_patches']} patches)")

    print(f'\nScores written to {out} (lower score = better aligned).\n')
