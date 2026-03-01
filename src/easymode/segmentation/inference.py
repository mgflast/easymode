import os, glob, time, multiprocessing, psutil
import tensorflow as tf
import gc
from tensorflow.keras import mixed_precision
import mrcfile
import numpy as np
from easymode.core.distribution import get_model, load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

TILE_SIZE = [128, 256, 256]
OVERLAP = [32, 32, 32]
MAX_CHUNK_SIZE = 1


def tile_volume(volume, patch_size, overlap):
    (pz, py, px) = patch_size; (oz, oy, ox) = overlap
    d, h, w = volume.shape
    sz, sy, sx = pz - 2*oz, py - 2*oy, px - 2*ox

    z_boxes = max(1, (d + sz - 1) // sz)
    y_boxes = max(1, (h + sy - 1) // sy)
    x_boxes = max(1, (w + sx - 1) // sx)

    tiles, positions = [], []
    for zi in range(z_boxes):
        for yi in range(y_boxes):
            for xi in range(x_boxes):
                z_start = zi*sz - oz; y_start = yi*sy - oy; x_start = xi*sx - ox
                vz0 = max(0, z_start); vy0 = max(0, y_start); vx0 = max(0, x_start)
                vz1 = min(d, z_start + pz); vy1 = min(h, y_start + py); vx1 = min(w, x_start + px)
                extracted = volume[vz0:vz1, vy0:vy1, vx0:vx1]

                tile = np.zeros((pz, py, px), dtype=volume.dtype)
                tz0 = vz0 - z_start; ty0 = vy0 - y_start; tx0 = vx0 - x_start
                tile[tz0:tz0+extracted.shape[0], ty0:ty0+extracted.shape[1], tx0:tx0+extracted.shape[2]] = extracted
                tiles.append(tile)
                positions.append((zi*sz, yi*sy, xi*sx))

    tiles = np.expand_dims(np.array(tiles), axis=-1)
    return tiles, positions, volume.shape

def detile_volume(segmented_tiles, positions, original_shape, patch_size, overlap):
    (pz, py, px) = patch_size; (oz, oy, ox) = overlap
    d, h, w = original_shape
    sz, sy, sx = pz - 2*oz, py - 2*oy, px - 2*ox

    out = np.zeros((d, h, w), dtype=np.float32)
    wgt = np.zeros((d, h, w), dtype=np.float32)
    if segmented_tiles.ndim == 5: segmented_tiles = segmented_tiles.squeeze(-1)

    for tile, (z_pos, y_pos, x_pos) in zip(segmented_tiles, positions):
        center = tile[oz:oz+sz, oy:oy+sy, ox:ox+sx]
        z_end = min(z_pos+sz, d); y_end = min(y_pos+sy, h); x_end = min(x_pos+sx, w)
        az, ay, ax = z_end - z_pos, y_end - y_pos, x_end - x_pos
        out[z_pos:z_end, y_pos:y_end, x_pos:x_end] += center[:az, :ay, :ax]
        wgt[z_pos:z_end, y_pos:y_end, x_pos:x_end] += 1.0

    wgt[wgt == 0] = 1.0
    return out / wgt

def _segment_tile_list(tiles, model, batch_size=8, max_chunk_size=MAX_CHUNK_SIZE):
    num_tiles = len(tiles)
    segmented_tiles = []
    for i in range(0, num_tiles, max_chunk_size):
        chunk_end = min(i + max_chunk_size, num_tiles)
        chunk = tiles[i:chunk_end]

        try:
            chunk_result = model.predict(chunk, verbose=0, batch_size=batch_size)
            segmented_tiles.extend(chunk_result)

        except tf.errors.ResourceExhaustedError:
            print(f"Memory error with chunk size {len(chunk)}, falling back to smaller chunks")
            fallback_chunk_size = max(1, len(chunk) // 4)
            for j in range(i, chunk_end, fallback_chunk_size):
                small_chunk = tiles[j:min(j + fallback_chunk_size, chunk_end)]
                small_result = model.predict(small_chunk, verbose=0, batch_size=batch_size)
                segmented_tiles.extend(small_result)

    return segmented_tiles


def _segment_tomogram_instance(volume, model, batch_size, tile_size, overlap):
    tiles, positions, original_shape = tile_volume(volume, tile_size, overlap)
    segmented_tiles = _segment_tile_list(tiles, model, batch_size=batch_size, max_chunk_size=MAX_CHUNK_SIZE)
    segmented_tiles = np.array(segmented_tiles)
    segmented_volume = detile_volume(segmented_tiles, positions, original_shape, tile_size, overlap)

    tf.keras.backend.clear_session()
    gc.collect()
    return segmented_volume.astype(np.float32)

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


def segment_tomogram(model, tomogram_path, tta=1, batch_size=2, model_apix=10.0, input_apix=None, model_apix_z=None):
    global TILE_SIZE, OVERLAP

    # model_apix_z=None means isotropic model (apix_z == apix_xy)
    if model_apix_z is None:
        model_apix_z = model_apix

    # load tomo & read pixel size
    with mrcfile.open(tomogram_path) as m:
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

    rescaled = False
    if abs(scale_xy - 1.0) > 0.05 or abs(scale_z - 1.0) > 0.05:
        from scipy.ndimage import zoom
        volume = zoom(volume, (scale_z, scale_xy, scale_xy), order=1)
        rescaled = True


    # preprocess: normalize & pad
    _j, _k, _l = volume.shape
    _k_margin = min(int(0.2*_k), 64)
    _l_margin = min(int(0.2*_l), 64)
    volume -= np.mean(volume[:, _k_margin:-_k_margin, _l_margin:-_l_margin])
    volume /= np.std(volume[:, _k_margin:-_k_margin, _l_margin:-_l_margin]) + 1e-7

    volume, padding = _pad_volume(volume)
    segmented_volume = np.zeros((oj, ok, ol), dtype=np.float32)

    TILE_SIZE = (min(256, volume.shape[0]), min(256, volume.shape[1]), min(256, volume.shape[2]))
    OVERLAP[0] = 0 if TILE_SIZE[0] == volume.shape[0] else 48
    OVERLAP[1] = 0 if TILE_SIZE[1] == volume.shape[1] else 48
    OVERLAP[2] = 0 if TILE_SIZE[2] == volume.shape[2] else 48

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
        segmented_tta_vol = _segment_tomogram_instance(tta_vol, model, batch_size, TILE_SIZE, OVERLAP)
        segmented_tta_vol = np.rot90(segmented_tta_vol, k=-2 * k_yz[j], axes=(0, 1))
        segmented_tta_vol = segmented_tta_vol if not k_fx[j] else np.flip(segmented_tta_vol, axis=1)
        segmented_tta_vol = np.rot90(segmented_tta_vol, k=-k_xy[j], axes=(1, 2))

        # remove padding per-tta instance
        (j0, j1), (k0, k1), (l0, l1) = padding
        segmented_tta_vol = segmented_tta_vol[j0:segmented_tta_vol.shape[0] - j1, k0:segmented_tta_vol.shape[1] - k1, l0:segmented_tta_vol.shape[2] - l1]

        if rescaled:
            from scipy.ndimage import zoom
            sj, sk, sl = segmented_tta_vol.shape
            segmented_tta_vol = zoom(segmented_tta_vol, (oj / sj, ok / sk, ol / sl), order=1)
            segmented_tta_vol = segmented_tta_vol[:oj, :ok, :ol]

        segmented_volume += segmented_tta_vol
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

def segmentation_thread(tomogram_list, model_path, feature, output_dir, gpu, batch_size, tta, overwrite, data_format, model_apix, model_apix_z=None, input_apix=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)

    process_start_time = psutil.Process().create_time()

    print(f'GPU {gpu} - loading model ({model_path}).')
    model = load_model(model_path)

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

            segmented_volume, segmented_volume_apix = segment_tomogram(model, tomogram_path, tta, batch_size, model_apix, input_apix, model_apix_z)

            save_mrc(segmented_volume, output_file, data_format, segmented_volume_apix)

            etc = time.strftime('%H:%M:%S', time.gmtime((time.time() - process_start_time) / j * (len(tomogram_list) - j)))
            print(f"{j}/{len(tomogram_list)} (on GPU {gpu}) - {feature} - {os.path.basename(tomogram_path)} - etc {etc}")
        except Exception as e:
            if wrote_temporary:
                os.remove(output_file)
            print(f"{j}/{len(tomogram_list)} (on GPU {gpu}) - {feature} - {os.path.basename(tomogram_path)} - ERROR: {e}")

def dispatch_segment( feature, data_directory, output_directory, tta=1, batch_size=8, overwrite=False, data_format='int8', gpus=None, data_apix=None ):
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
    )

    # Collect tomograms from all patterns / paths
    tomograms = []
    for p in patterns:
        if os.path.isdir(p):
            matches = glob.glob(os.path.join(p, '*.mrc'))
        else:
            matches = glob.glob(p)
        tomograms.extend(matches)

    tomograms = [f for f in sorted(set(tomograms)) if os.path.splitext(f)[-1] == '.mrc']

    print(f'Found {len(tomograms)} tomograms to segment. \n')

    if len(tomograms) == 0:
        return

    model_path, metadata = get_model(feature)
    if model_path is None:
        print(f'Could not find model for {feature}! Exiting.')
        return
    model_apix = metadata["apix"]
    model_apix_z = metadata.get("apix_z", None)  # None means isotropic

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
                data_apix
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
        for path in glob.glob(os.path.join(output_directory, f'*__{feature}.mrc')):
            if os.path.getsize(path) < 10_000:
                os.remove(path)
        return

    print()
    print(f'\033[92mSegmentation finished!\033[0m')
    print(f"\033[96mPlease note that you can help improve easymode by reporting model failures with 'easymode report'\033[0m")
    print()

