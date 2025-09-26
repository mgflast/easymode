import os, glob, time, multiprocessing, psutil
import tensorflow as tf
import gc
from tensorflow.keras import mixed_precision
import mrcfile
import numpy as np
from easymode.core.distribution import cache_model, load_model

TILE_SIZE = 128
OVERLAP = 16
MAX_CHUNK_SIZE = 64

def tile_volume(volume, patch_size=TILE_SIZE, overlap=OVERLAP):
    d, h, w = volume.shape
    stride = patch_size - 2 * overlap

    z_boxes = max(1, (d + stride - 1) // stride)
    y_boxes = max(1, (h + stride - 1) // stride)
    x_boxes = max(1, (w + stride - 1) // stride)

    tiles = []
    positions = []

    for z_idx in range(z_boxes):
        for y_idx in range(y_boxes):
            for x_idx in range(x_boxes):
                z_start = z_idx * stride - overlap
                y_start = y_idx * stride - overlap
                x_start = x_idx * stride - overlap

                vol_z_start = max(0, z_start)
                vol_y_start = max(0, y_start)
                vol_x_start = max(0, x_start)

                vol_z_end = min(d, z_start + patch_size)
                vol_y_end = min(h, y_start + patch_size)
                vol_x_end = min(w, x_start + patch_size)

                extracted = volume[vol_z_start:vol_z_end, vol_y_start:vol_y_end, vol_x_start:vol_x_end]

                tile = np.zeros((patch_size, patch_size, patch_size), dtype=volume.dtype)

                tile_z_start = vol_z_start - z_start
                tile_y_start = vol_y_start - y_start
                tile_x_start = vol_x_start - x_start

                tile[tile_z_start:tile_z_start + extracted.shape[0],
                tile_y_start:tile_y_start + extracted.shape[1],
                tile_x_start:tile_x_start + extracted.shape[2]] = extracted

                tiles.append(tile)
                positions.append((z_idx * stride, y_idx * stride, x_idx * stride))

    tiles = np.array(tiles)
    tiles = np.expand_dims(tiles, axis=-1)

    return tiles, positions, volume.shape

def detile_volume(denoised_tiles, positions, original_shape, patch_size=TILE_SIZE, overlap=OVERLAP):
    d, h, w = original_shape
    output_volume = np.zeros((d, h, w), dtype=np.float32)
    stride = patch_size - 2 * overlap

    if denoised_tiles.ndim == 5:
        denoised_tiles = denoised_tiles.squeeze(-1)

    for tile, (z_pos, y_pos, x_pos) in zip(denoised_tiles, positions):
        center_region = tile[overlap:overlap + stride, overlap:overlap + stride, overlap:overlap + stride]

        z_end = min(z_pos + stride, d)
        y_end = min(y_pos + stride, h)
        x_end = min(x_pos + stride, w)

        actual_z = z_end - z_pos
        actual_y = y_end - y_pos
        actual_x = x_end - x_pos

        output_volume[z_pos:z_end, y_pos:y_end, x_pos:x_end] = center_region[:actual_z, :actual_y, :actual_x]

    return output_volume

def _denoise_tile_list(tiles, model, batch_size=8, max_chunk_size=MAX_CHUNK_SIZE):
    num_tiles = len(tiles)
    denoised_tiles = []
    for i in range(0, num_tiles, max_chunk_size):
        chunk_end = min(i + max_chunk_size, num_tiles)
        chunk = tiles[i:chunk_end]

        try:
            chunk_result = model.predict(chunk, verbose=0, batch_size=batch_size)
            denoised_tiles.extend(chunk_result)

        except tf.errors.ResourceExhaustedError:
            print(f"Memory error with chunk size {len(chunk)}, falling back to smaller chunks")
            fallback_chunk_size = max(1, len(chunk) // 4)
            for j in range(i, chunk_end, fallback_chunk_size):
                small_chunk = tiles[j:min(j + fallback_chunk_size, chunk_end)]
                small_result = model.predict(small_chunk, verbose=0, batch_size=batch_size)
                denoised_tiles.extend(small_result)

    return denoised_tiles

def _denoise_tomogram_instance(volume, model, batch_size):
    tiles, positions, original_shape = tile_volume(volume)
    denoised_tiles = _denoise_tile_list(tiles, model, batch_size=batch_size, max_chunk_size=MAX_CHUNK_SIZE)
    denoised_tiles = np.array(denoised_tiles)
    denoised_volume = detile_volume(denoised_tiles, positions, original_shape)

    tf.keras.backend.clear_session()
    gc.collect()
    return denoised_volume.astype(np.float32)


def denoise_tomogram(model, tomogram_path, tta=1, batch_size=2, iter=1):
    volume = mrcfile.read(tomogram_path).astype(np.float32)
    volume = np.pad(volume, pad_width=16, mode='reflect')

    # Below: all 16 combinations of right angle rotations and flips that respect the anisotropy of the data.
    k_xy = [0, 2, 2, 0, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 3]
    k_fx = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    k_yz = [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]

    for i in range(iter):
        volume -= np.mean(volume)
        volume /= np.std(volume) + 1e-6
        denoised_volume = np.zeros_like(volume)
        for j in range(tta):
            tta_vol = volume.copy()
            tta_vol = np.rot90(tta_vol, k=k_xy[j], axes=(1, 2))
            tta_vol = tta_vol if not k_fx[j] else np.flip(tta_vol, axis=1)
            tta_vol = np.rot90(tta_vol, k=2 * k_yz[j], axes=(0, 1))
            denoised_tta_vol = _denoise_tomogram_instance(tta_vol, model, batch_size)
            denoised_tta_vol = np.rot90(denoised_tta_vol, k=-2 * k_yz[j], axes=(0, 1))
            denoised_tta_vol = denoised_tta_vol if not k_fx[j] else np.flip(denoised_tta_vol, axis=1)
            denoised_tta_vol = np.rot90(denoised_tta_vol, k=-k_xy[j], axes=(1, 2))
            denoised_volume += denoised_tta_vol
        denoised_volume /= tta
        volume = denoised_volume

    volume = volume[16:-16, 16:-16, 16:-16]
    return volume

def save_mrc(pxd, path, data_format, voxel_size=10.0):
    if data_format == 'float32':
        pxd = pxd.astype(np.float32)
    # TODO: float16
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(pxd)
        m.voxel_size = voxel_size

def denoiser_thread(mode, tomogram_list, model_path, output_dir, gpu, batch_size, tta, overwrite, iter):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ['TF_DISABLE_MKL'] = '1'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    #mixed_precision.set_global_policy('mixed_float16')

    process_start_time = psutil.Process().create_time()
    model = load_model(model_path)

    for j, tomogram_tuple in enumerate(tomogram_list, start=1):
        tomo_name = os.path.splitext(os.path.basename(tomogram_tuple[0]))[0]
        output_file = os.path.join(output_dir, f"{tomo_name}.mrc")
        wrote_temporary = False
        try:
            if os.path.exists(output_file):
                file_age = os.path.getmtime(output_file)
                if not overwrite or file_age > process_start_time - 60:
                    continue

            with mrcfile.new(output_file, overwrite=True) as m:
                m.set_data(-1.0 * np.ones((10, 10, 10), dtype=np.float32))
                wrote_temporary = True


            if mode=='splits':
                denoised_volume = (denoise_tomogram(model, tomogram_tuple[0], tta, batch_size) + denoise_tomogram(model, tomogram_tuple[1], tta, batch_size)) / 2.0
            else:
                denoised_volume = denoise_tomogram(model, tomogram_tuple[0], tta, batch_size, iter=iter)

            save_mrc(denoised_volume, output_file, data_format='float32')

            etc = time.strftime('%H:%M:%S', time.gmtime((time.time() - process_start_time) / j * (len(tomogram_list) - j)))
            print(f"{j}/{len(tomogram_list)} (on GPU {gpu}) - {os.path.basename(output_file)} - etc: {etc}")
        except Exception as e:
            if wrote_temporary:
                os.remove(output_file)
            print(f"{j}/{len(tomogram_list)} (on GPU {gpu}) - {os.path.basename(output_file)} - ERROR: {e}")

def dispatch(input_directory, output_directory, mode='splits', tta=1, batch_size=8, overwrite=False, iter=1, gpus="0"):
    if output_directory == input_directory:
        print("Please choose an output directory that is different from the input directory - we dont want to overwrite your original volumes.")
        exit()

    gpus = [int(g) for g in gpus.split(',') if g.strip().isdigit()]

    print(f'easymode denoise\n'
          f'mode: {mode}\n'
          f'data_directory: {input_directory}\n'
          f'output_directory: {output_directory}\n'
          f'gpus: {gpus}\n'
          f'tta: {tta}\n'
          f'overwrite: {overwrite}\n'
          f'batch_size: {batch_size}\n')

    if mode == 'direct':
        tomograms = sorted(glob.glob(os.path.join(input_directory, '*.mrc')))
        tomograms = list(zip(tomograms, tomograms))
    else:
        tomograms_evn = sorted(glob.glob(os.path.join(input_directory, 'even', '*.mrc')))
        tomograms_odd = [os.path.join(input_directory, 'odd', f'{os.path.basename(f)}') for f in tomograms_evn]
        for t in tomograms_odd:
            if not os.path.exists(t):
                raise FileNotFoundError(f"Could not find matching odd tomogram for {t}")

        tomograms = list(zip(tomograms_evn, tomograms_odd))

    print(f'Found {len(tomograms)} tomograms to denoise in {input_directory}.')

    model_path = cache_model('n2n_splits' if mode=='splits' else 'n2n_direct')

    os.makedirs(output_directory, exist_ok=True)

    processes = []
    for gpu in gpus:
        p = multiprocessing.Process(target=denoiser_thread, args=(mode, tomograms, model_path, output_directory, gpu, batch_size, tta, overwrite, iter))
        processes.append(p)
        p.start()
        time.sleep(2)

    for p in processes:
        p.join()






