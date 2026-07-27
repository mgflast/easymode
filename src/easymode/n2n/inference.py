import os, glob, time, multiprocessing, psutil
import tensorflow as tf
import gc
import mrcfile
import numpy as np
from easymode.core.distribution import get_model, load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

TILE_SIZE = 160
OVERLAP = 32

def _denoise_tomogram_instance(volume, model, batch_size):
    # Stream one tile at a time: the tomogram stays in RAM but tiles/predictions
    # are never all materialised at once. Geometry, zero-padding, central-core crop
    # and hard placement are identical to the previous tile/detile behaviour (the
    # cores stride by their own size, so they never overlap). batch_size is ignored;
    # denoising always runs one tile per model call.
    patch_size, overlap = TILE_SIZE, OVERLAP
    stride = patch_size - 2 * overlap
    d, h, w = volume.shape

    z_boxes = max(1, (d + stride - 1) // stride)
    y_boxes = max(1, (h + stride - 1) // stride)
    x_boxes = max(1, (w + stride - 1) // stride)

    output_volume = np.zeros((d, h, w), dtype=np.float32)
    for z_idx in range(z_boxes):
        for y_idx in range(y_boxes):
            for x_idx in range(x_boxes):
                z_pos, y_pos, x_pos = z_idx * stride, y_idx * stride, x_idx * stride
                z_start, y_start, x_start = z_pos - overlap, y_pos - overlap, x_pos - overlap

                vz0, vy0, vx0 = max(0, z_start), max(0, y_start), max(0, x_start)
                vz1 = min(d, z_start + patch_size)
                vy1 = min(h, y_start + patch_size)
                vx1 = min(w, x_start + patch_size)

                tile = np.zeros((patch_size, patch_size, patch_size), dtype=volume.dtype)
                tz0, ty0, tx0 = vz0 - z_start, vy0 - y_start, vx0 - x_start
                extracted = volume[vz0:vz1, vy0:vy1, vx0:vx1]
                tile[tz0:tz0 + extracted.shape[0], ty0:ty0 + extracted.shape[1], tx0:tx0 + extracted.shape[2]] = extracted

                prediction = model(tile[None, ..., None], training=False).numpy()[0, ..., 0]

                center = prediction[overlap:overlap + stride, overlap:overlap + stride, overlap:overlap + stride]
                z_end = min(z_pos + stride, d); y_end = min(y_pos + stride, h); x_end = min(x_pos + stride, w)
                output_volume[z_pos:z_end, y_pos:y_end, x_pos:x_end] = center[:z_end - z_pos, :y_end - y_pos, :x_end - x_pos]

    tf.keras.backend.clear_session()
    gc.collect()
    return output_volume.astype(np.float32)


def denoise_tomogram(model, tomogram_path, tta=1, batch_size=2, iter=1):
    with mrcfile.open(tomogram_path) as m:
        volume = m.data.astype(np.float32)
        volume_apix = float(m.voxel_size.x)
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
    return volume, volume_apix

def save_mrc(pxd, path, data_format, voxel_size=10.0):
    if data_format == 'float32':
        pxd = pxd.astype(np.float32)
    # TODO: float16
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(pxd)
        m.voxel_size = voxel_size

METHOD_TO_WEIGHTS = {
    'n2n': 'n2n_direct',                          # noise2noise direct-input student (the deployed default)
    'ddw': 'ddw_direct',                          # distilled DeepDeWedge student (new; also fills the wedge)
    'iso': 'iso_direct',                          # distilled IsoNet2 student (also fills the wedge; not distributed yet -- use --weights)
}


def denoiser_thread(tomogram_list, model_path, output_dir, gpu, batch_size, tta, overwrite, iter):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)

    process_start_time = psutil.Process().create_time()

    print(f'GPU {gpu} - loading model ({model_path}).')
    model = load_model(model_path)

    print(f'GPU {gpu} - starting inference.')

    for j, tomo_path in enumerate(tomogram_list, start=1):
        tomo_name = os.path.splitext(os.path.basename(tomo_path))[0]
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

            denoised_volume, volume_apix = denoise_tomogram(model, tomo_path, tta, batch_size, iter=iter)
            save_mrc(denoised_volume, output_file, data_format='float32', voxel_size=volume_apix)

            etc = time.strftime('%H:%M:%S', time.gmtime((time.time() - process_start_time) / j * (len(tomogram_list) - j)))
            print(f"{j}/{len(tomogram_list)} (on GPU {gpu}) - {os.path.basename(output_file)} - etc: {etc}")
        except Exception as e:
            if wrote_temporary:
                os.remove(output_file)
            print(f"{j}/{len(tomogram_list)} (on GPU {gpu}) - {os.path.basename(output_file)} - ERROR: {e}")


def dispatch(input_directory, output_directory, method='n2n', tta=1, batch_size=8, overwrite=False, iter=1, gpus="0"):
    if output_directory == input_directory:
        print("Please choose an output directory that is different from the input directory - we dont want to overwrite your original volumes.")
        exit()

    if method not in METHOD_TO_WEIGHTS:
        raise ValueError(f"unknown denoising method {method!r}; available: {sorted(METHOD_TO_WEIGHTS)}")

    if gpus is None:
        gpus = list(range(0, len(tf.config.list_physical_devices('GPU'))))
    else:
        gpus = [int(g) for g in gpus.split(',') if g.strip().isdigit()]

    if len(gpus) == 0:
        print("\033[93m" + "warning: no GPUs detected. processing will continue, but using CPUs only!" + "\033[0m")
        gpus = [-1]

    print(f'easymode denoise\n'
          f'method: {method}\n'
          f'data_directory: {input_directory}\n'
          f'output_directory: {output_directory}\n'
          f'gpus: {gpus}\n'
          f'tta: {tta}\n'
          f'overwrite: {overwrite}\n'
          f'batch_size: {batch_size}\n')

    tomograms = sorted(glob.glob(os.path.join(input_directory, '*.mrc')))
    print(f'Found {len(tomograms)} tomograms to denoise in {input_directory}.')

    model_path = get_model(METHOD_TO_WEIGHTS[method])[0]

    os.makedirs(output_directory, exist_ok=True)

    multiprocessing.set_start_method('spawn', force=True)

    processes = []
    for gpu in gpus:
        p = multiprocessing.Process(target=denoiser_thread, args=(tomograms, model_path, output_directory, gpu, batch_size, tta, overwrite, iter))
        processes.append(p)
        p.start()
        time.sleep(2)

    for p in processes:
        p.join()






