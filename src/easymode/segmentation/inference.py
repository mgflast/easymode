import os, sys, glob, time, atexit, signal, multiprocessing, psutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF C++ INFO and WARNING before import
import tensorflow as tf
import gc
from tensorflow.keras import mixed_precision
import mrcfile
import numpy as np
from easymode.core.distribution import get_model, load_model
from easymode.segmentation.normalization import global_stats, NORM_GLOBAL_STD, NORM_GLOBAL_MAD, NORM_LOCAL
from skimage.transform import resize

tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

DEFAULT_TILE_SIZE = (160, 256, 256)  # (Z, Y, X); capped per-axis to the volume dims at tiling time
DEFAULT_OVERLAP = 24


_PLACEHOLDER_BYTES = 8192   # the 10x10x10 claim file is ~5 kB; a real segmentation is megabytes
_claimed_outputs = set()    # claim files THIS process wrote and has not yet overwritten with real data


def _claim_output(path):
    _claimed_outputs.add(path)


def _release_output(path):
    _claimed_outputs.discard(path)


def _drop_claimed_outputs():
    """Remove the claim files this process is still holding, on cancellation. Only its own: sibling
    jobs on other nodes segment into the same directory, and their claim files mark tomograms they
    are still working on. The size check keeps a just-finished real segmentation safe."""
    for path in list(_claimed_outputs):
        _claimed_outputs.discard(path)
        try:
            if os.path.getsize(path) <= _PLACEHOLDER_BYTES:
                os.remove(path)
        except OSError:
            pass


def _install_claim_cleanup():
    atexit.register(_drop_claimed_outputs)
    try:   # terminate() / scancel would otherwise skip atexit and leak the claim file
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    except (ValueError, OSError):
        pass


def _max_overlap(tile):
    # tiling strides by (tile - 2*overlap), so the overlap must stay below tile/2 or the stride
    # goes <= 0 and tiling breaks. Cap at tile//3 to keep the stride at a workable >= tile/3.
    return max(0, int(tile) // 3)


def _segment_tomogram_instance(volume, model, batch_size, tile_size, overlap, normalize_tiles=False):
    # Stream one tile at a time: the tomogram stays in RAM but tiles/predictions
    # are never all materialised at once. Geometry, zero-padding, central-core crop
    # and hard placement are identical to the previous tile/detile behaviour (the
    # cores stride by their own size, so they never overlap). batch_size is ignored;
    # the 3-D path always runs one tile per model call.
    (pz, py, px) = tile_size
    (oz, oy, ox) = overlap
    d, h, w = volume.shape
    sz, sy, sx = pz - 2*oz, py - 2*oy, px - 2*ox

    z_boxes = max(1, (d + sz - 1) // sz)
    y_boxes = max(1, (h + sy - 1) // sy)
    x_boxes = max(1, (w + sx - 1) // sx)

    out = np.zeros((d, h, w), dtype=np.float32)
    for zi in range(z_boxes):
        for yi in range(y_boxes):
            for xi in range(x_boxes):
                z_pos, y_pos, x_pos = zi*sz, yi*sy, xi*sx
                z_start, y_start, x_start = z_pos - oz, y_pos - oy, x_pos - ox
                vz0, vy0, vx0 = max(0, z_start), max(0, y_start), max(0, x_start)
                vz1 = min(d, z_start + pz); vy1 = min(h, y_start + py); vx1 = min(w, x_start + px)

                tile = np.zeros((pz, py, px), dtype=volume.dtype)
                tz0, ty0, tx0 = vz0 - z_start, vy0 - y_start, vx0 - x_start
                extracted = volume[vz0:vz1, vy0:vy1, vx0:vx1]
                tile[tz0:tz0+extracted.shape[0], ty0:ty0+extracted.shape[1], tx0:tx0+extracted.shape[2]] = extracted

                if normalize_tiles:
                    _c, _s = global_stats(tile)
                    tile = (tile - _c) / _s

                prediction = model(tile[None, ..., None], training=False).numpy()[0, ..., 0]

                center = prediction[oz:oz+sz, oy:oy+sy, ox:ox+sx]
                z_end = min(z_pos+sz, d); y_end = min(y_pos+sy, h); x_end = min(x_pos+sx, w)
                out[z_pos:z_end, y_pos:y_end, x_pos:x_end] = center[:z_end-z_pos, :y_end-y_pos, :x_end-x_pos]

    return out.astype(np.float32)

def _pad_volume(volume, min_pad=16, div=32):
    j, k, l = volume.shape
    pads = []
    for n in (j, k, l):
        total_pad = max(2*min_pad, ((n + 2*min_pad + div - 1)//div)*div - n)
        before = total_pad // 2
        after = total_pad - before
        pads.append((before, after))
    padded = np.pad(volume, pads, mode='reflect')
    return padded, tuple(pads)


def segment_tomogram(model, tomogram_path, tta=1, batch_size=2, model_apix=10.0, input_apix=None, model_apix_z=None, use_depth=1.0, xy_margin=0, tile_size=None, overlap=None, normalization=NORM_GLOBAL_STD):
    if tile_size is None:
        tile_size = DEFAULT_TILE_SIZE
    if overlap is None:
        overlap = DEFAULT_OVERLAP

    # model_apix_z=None means isotropic model (apix_z == apix_xy)
    if model_apix_z is None:
        model_apix_z = model_apix

    # load tomo & read pixel size
    with mrcfile.open(tomogram_path, permissive=True) as m:
        volume = m.data.astype(np.float32)
        volume_apix = m.voxel_size.x
        oj, ok, ol = volume.shape
        if volume_apix <= 1.0 and input_apix is None:
            print(f'warning: {tomogram_path} header lists voxel size as 1.0 A/px, which is probably incorrect. We assume it is really 10.0 Å/px.')
            volume_apix = 10.0

    # compute per-axis scale factors (model may have been trained with XY binning but no Z binning)
    if input_apix is None:
        scale_xy = float(volume_apix) / float(model_apix)
        scale_z = float(volume_apix) / float(model_apix_z)
    elif input_apix == 0.0:
        scale_xy = 1.0
        scale_z = 1.0
    else:
        scale_xy = float(input_apix) / float(model_apix)
        scale_z = float(input_apix) / float(model_apix_z)

    # New (global_mad) models normalize at NATIVE resolution, BEFORE rescaling, so extract
    # and inference measure the statistic identically. Scheme comes from model metadata.
    if normalization == NORM_GLOBAL_MAD:
        _center, _scale = global_stats(volume, method=normalization)
        volume -= _center
        volume /= _scale

    # preprocess: scale to model apix.
    rescaled = False
    if abs(scale_xy - 1.0) > 0.05 or abs(scale_z - 1.0) > 0.05:
        new_size = [int(np.round(scale_z * oj)), int(np.round(scale_xy * ok)), int(np.round(scale_xy * ol))]
        volume = resize(volume, new_size, order=3, anti_aliasing=True).astype(np.float32)
        rescaled = True

    # Legacy (global_std) models keep the pre-2026 behaviour: mean/std normalization AFTER
    # rescale (the stat is now sampled rather than full-volume, a <0.5% input-scale change
    # the model's input norm absorbs). local models are normalized per tile instead, below.
    if normalization not in (NORM_GLOBAL_MAD, NORM_LOCAL):
        _center, _scale = global_stats(volume, method=normalization)
        volume -= _center
        volume /= _scale

    # Compute active Z range (in current/possibly-rescaled volume coords and in original coords)
    # Skip Z cropping if the active region would be too small (fewer than 32 slices)
    n_z = volume.shape[0]
    z_margin = int(n_z * (1.0 - use_depth) / 2) if n_z * use_depth >= 32 else 0
    z_start_r, z_end_r = z_margin, n_z - z_margin
    z_start = int(oj * (1.0 - use_depth) / 2) if z_margin > 0 else 0
    z_end = oj - z_start

    # Compute XY margin in current (possibly-rescaled) coords; skip if the tomogram is too small
    xy_margin_r = int(xy_margin * (volume.shape[1] / ok)) if xy_margin > 0 and ok >= xy_margin * 5 and ol >= xy_margin * 5 else 0
    xy_margin = xy_margin if xy_margin_r > 0 else 0

    volume = volume[z_start_r:z_end_r,
                    xy_margin_r:volume.shape[1] - xy_margin_r if xy_margin_r else volume.shape[1],
                    xy_margin_r:volume.shape[2] - xy_margin_r if xy_margin_r else volume.shape[2]]

    volume, padding = _pad_volume(volume)
    segmented_volume = np.zeros((oj, ok, ol), dtype=np.float32)

    tile = tuple(int(min(t, s)) for t, s in zip(tile_size, volume.shape))
    tile_overlap = tuple(0 if tile[i] == volume.shape[i] else min(overlap, _max_overlap(tile[i])) for i in range(3))

    # Below: all 16 combinations of 90-degree rotations and flips that respect the anisotropy of the data.
    k_xy = [0, 2, 2, 0, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 3]
    k_fx = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    k_yz = [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]

    # inference loop w tta
    for j in range(tta):
        tta_vol = volume.copy()
        tta_vol = np.rot90(tta_vol, k=k_xy[j], axes=(1, 2))
        tta_vol = tta_vol if not k_fx[j] else np.flip(tta_vol, axis=1)
        tta_vol = np.rot90(tta_vol, k=2 * k_yz[j], axes=(0, 1))
        segmented_tta_vol = _segment_tomogram_instance(tta_vol, model, batch_size, tile, tile_overlap,
                                                       normalize_tiles=normalization == NORM_LOCAL)
        segmented_tta_vol = np.rot90(segmented_tta_vol, k=-2 * k_yz[j], axes=(0, 1))
        segmented_tta_vol = segmented_tta_vol if not k_fx[j] else np.flip(segmented_tta_vol, axis=1)
        segmented_tta_vol = np.rot90(segmented_tta_vol, k=-k_xy[j], axes=(1, 2))

        # remove padding per-tta instance
        (j0, j1), (k0, k1), (l0, l1) = padding
        segmented_tta_vol = segmented_tta_vol[j0:segmented_tta_vol.shape[0] - j1, k0:segmented_tta_vol.shape[1] - k1, l0:segmented_tta_vol.shape[2] - l1]

        if rescaled:
            from scipy.ndimage import zoom
            sj, sk, sl = segmented_tta_vol.shape
            segmented_tta_vol = resize(segmented_tta_vol, (z_end - z_start, ok - 2 * xy_margin, ol - 2 * xy_margin), order=1)
            segmented_tta_vol = segmented_tta_vol[:z_end - z_start, :ok - 2 * xy_margin, :ol - 2 * xy_margin]

        segmented_volume[z_start:z_end, xy_margin:ok - xy_margin if xy_margin else ok, xy_margin:ol - xy_margin if xy_margin else ol] += segmented_tta_vol
    segmented_volume /= tta

    return segmented_volume, volume_apix


def save_mrc(pxd, path, data_format, voxel_size=10.0):
    if data_format == 'float32':
        pxd = pxd.astype(np.float32)
    elif data_format == 'uint16':
        pxd = (pxd * 255).astype(np.uint16)  # scaling to [0, 255] because that's what we're used to in Ais
    elif data_format == 'int8':
        pxd = (pxd * 127).astype(np.int8)
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(pxd)
        m.voxel_size = voxel_size

def segmentation_thread(tomogram_list, model_path, feature, output_dir, gpu, batch_size, tta, overwrite, data_format, model_apix, model_apix_z=None, input_apix=None, use_depth=1.0, xy_margin=0, tile_size=None, overlap=None, normalization=NORM_GLOBAL_STD):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _install_claim_cleanup()

    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)

    print(f'GPU {gpu} - loading model ({model_path}).')
    model = load_model(model_path)

    process_start_time = psutil.Process().create_time()
    print(f'GPU {gpu} - starting inference.')

    for j, tomogram_path in enumerate(tomogram_list, 1):
        tomo_name = os.path.splitext(os.path.basename(tomogram_path))[0]
        output_file = os.path.join(output_dir, f"{tomo_name}__{feature}.mrc")
        wrote_temporary = False
        try:
            if os.path.exists(output_file):
                temp_file_write_time = os.path.getmtime(output_file)
                if not overwrite or temp_file_write_time > process_start_time:
                    continue

            with mrcfile.new(output_file, overwrite=True) as m:
                m.set_data(-1.0 * np.ones((10, 10, 10), dtype=np.float32))
                m.voxel_size = 10.0
                wrote_temporary = True
            _claim_output(output_file)

            segmented_volume, segmented_volume_apix = segment_tomogram(model, tomogram_path, tta, batch_size, model_apix, input_apix, model_apix_z, use_depth, xy_margin, tile_size, overlap, normalization)

            save_mrc(segmented_volume, output_file, data_format, segmented_volume_apix)
            _release_output(output_file)

            elapsed = time.time() - process_start_time
            eta = time.strftime('%H:%M:%S', time.gmtime(elapsed / j * (len(tomogram_list) - j)))
            avg_secs = int(elapsed / j)
            per_tomo = f"{avg_secs // 60:02d}:{avg_secs % 60:02d}"
            print(f"{j}/{len(tomogram_list)} (on GPU {gpu}) - {feature} - {os.path.basename(tomogram_path)} - etc {eta} ({per_tomo} per tomo)")
        except Exception as e:
            if wrote_temporary:
                _release_output(output_file)
                os.remove(output_file)
            print(f"{j}/{len(tomogram_list)} (on GPU {gpu}) - {feature} - {os.path.basename(tomogram_path)} - ERROR: {e}")

def dispatch_segment(feature, data_directory, output_directory, tta=1, batch_size=8, overwrite=False, data_format='int8', gpus=None, data_apix=None, use_depth=1.0, xy_margin=0, tile_size=None, overlap=None, model_path=None):
    if tile_size is None:
        tile_size = DEFAULT_TILE_SIZE
    if overlap is None:
        overlap = DEFAULT_OVERLAP

    if any(t < 1 for t in tile_size):
        print(f'Tile size {tuple(tile_size)} must be positive in every dimension! Exiting.')
        return
    if overlap < 0:
        print(f'Overlap {overlap} cannot be negative! Exiting.')
        return

    capped = min(_max_overlap(t) for t in tile_size)
    if overlap > capped:
        print("\033[93m" + f'warning: an overlap of {overlap} is too large for a tile size of '
              f'{tuple(tile_size)} (the tiling stride would be <= 0). Reducing the overlap to {capped}. '
              f'Use a larger tile or a smaller overlap to avoid this.' + "\033[0m")
    if isinstance(data_directory, (list, tuple)):
        patterns = list(data_directory)
    else:
        patterns = [data_directory]

    if output_directory is None:
        output_directory = 'segmented'

    if gpus is None:
        gpus = list(range(0, len(tf.config.list_physical_devices('GPU'))))
    else:
        gpus = [int(g) for g in gpus.split(',') if g.strip().isdigit()]

    if len(gpus) == 0:
        print("\033[93m" + "warning: no GPUs detected. processing will continue, but using CPUs only!" + "\033[0m")
        gpus = [-1]

    print(
        f'easymode segment\n'
        f'feature: {feature}\n'
        f'data_patterns: {patterns}\n'
        f'output_directory: {output_directory}\n'
        f'output_format: {data_format}\n'
        f'gpus: {gpus}\n'
        f'tta: {tta}\n'
        f'overwrite: {overwrite}\n'
        f'batch_size: {batch_size}\n'
        f'tile_size: {tuple(tile_size)}\n'
        f'overlap: {overlap}\n'
    )

    # Collect tomograms from all patterns / paths
    tomograms = []
    for p in patterns:
        if p.endswith('.txt') and os.path.isfile(p):
            # .txt file with one tomogram path per line (e.g. a pom subset) (260331)
            with open(p) as f:
                for line in f:
                    entry = line.strip()
                    if entry and not entry.endswith('.mrc'):
                        entry += '.mrc'
                    if entry:
                        tomograms.append(entry)
        elif os.path.isdir(p):
            matches = glob.glob(os.path.join(p, '*.mrc'))
            tomograms.extend(matches)
        else:
            matches = glob.glob(p)
            tomograms.extend(matches)

    tomograms = [f for f in sorted(set(tomograms)) if os.path.splitext(f)[-1] == '.mrc']

    print(f'Found {len(tomograms)} tomograms to segment. \n')

    if len(tomograms) == 0:
        return

    if model_path is None:
        model_path, metadata = get_model(feature)
        if model_path is None:
            print(f'Could not find model for {feature}! Exiting.')
            return
    else:
        from easymode.core.distribution import read_local_metadata
        metadata = read_local_metadata(os.path.splitext(model_path)[0] + '.json')
        if metadata is None:
            print(f'No metadata found at {os.path.splitext(model_path)[0]}.json - a model needs its .json sidecar. Exiting.')
            return
    model_apix = metadata.get("apix", 10.0)
    model_apix_z = metadata.get("apix_z", None)  # None means isotropic
    model_norm = metadata.get("normalization", NORM_GLOBAL_STD)  # untagged/old models -> legacy std

    if model_apix_z is not None:
        print(f'Using model: {model_path}, inference at {model_apix} Å/px (XY) / {model_apix_z} Å/px (Z). \n')
    else:
        print(f'Using model: {model_path}, inference at {model_apix} Å/px. \n')

    os.makedirs(output_directory, exist_ok=True)

    multiprocessing.set_start_method('spawn', force=True)

    processes = []
    for gpu in gpus:
        p = multiprocessing.Process(
            target=segmentation_thread,
            args=(
                tomograms,
                model_path,
                feature,
                output_directory,
                gpu,
                batch_size,
                tta,
                overwrite,
                data_format,
                model_apix,
                model_apix_z,
                data_apix,
                use_depth,
                xy_margin,
                tile_size,
                overlap,
                model_norm
            )
        )
        processes.append(p)
        p.start()
        time.sleep(2)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        # each worker drops its own claim files (_drop_claimed_outputs); the small files left in
        # this directory by sibling jobs on other nodes are tomograms they are still segmenting.
        return

    print()
    print(f'\033[92mSegmentation finished!\033[0m')
    print(f"\033[96mPlease note that you can help improve easymode by reporting model failures with 'easymode report'\033[0m")
    print()

