import subprocess, glob, os, tarfile, json, tempfile, time, multiprocessing, psutil, starfile
import numpy as np
import mrcfile
from multiprocessing import cpu_count
from easymode.core.distribution import get_model


def _run(cmd, capture=False):
    print(f'\033[42m{cmd}\033[0m\n')
    ret = subprocess.run(cmd, shell=True, capture_output=capture, text=True if capture else None)
    if ret.returncode != 0:
        print(f'\033[91merror running {cmd}\033[0m')
        exit()
    return ret.stdout


def pick(data_directory, target, output_directory, threshold, spacing, size, binning=2, tomostar=True, filament=False, per_filament_star_file=False, filament_length=500, centroid=False, min_particles=0, rotation_per_sample=0.0):
    if output_directory is None:
        output_directory = f'coordinates/{target}'

    print(f'easymode pick\n'
          f'feature: {target}\n'
          f'filament_mode: {filament}\n'
          f'data_directory: {data_directory}\n'
          f'output_directory: {output_directory}\n'
          f'output_pattern: *__{target}_coords.star\n'
          f'threshold: {threshold}\n'
          f'spacing: {spacing} Å\n'
          f'size: {size} Å^3\n'
          f'binning: {binning}\n'
          f'n_processes: {cpu_count()}\n'
          f'rename to .tomostar: {tomostar}\n'
          f'per_filament_star_file: {per_filament_star_file}\n'
          f'filament_length: {filament_length} Å\n'
          f'rotation_per_sample: {rotation_per_sample} degrees\n'
          f'centroid: {centroid}\n')

    command = f'ais pick -t {target} -d {data_directory} -ou {output_directory} -threshold {threshold} -spacing {spacing} -size {size} -b {binning} -p {cpu_count()} -min-particles {min_particles}'
    if filament:
        command += f' -filament -length {filament_length} --twist {rotation_per_sample}'
    if centroid:
        command += ' -centroid'
    _run(command)

    if tomostar:  # rename the rlnMicrograph name to account for the Warp(Tools) .tomostar / _10.00Apx.mrc discrepancy
        files = glob.glob(f'{output_directory}/*__{target}_coords.star')
        n_particles = 0
        for j, f in enumerate(files):
            data = starfile.read(f)
            n_particles += len(data)
            tomo = os.path.basename(f.split('_10.00Apx')[0]) + '.tomostar'
            data["rlnMicrographName"] = tomo
            starfile.write({"particles": data}, f)

    if per_filament_star_file and filament:
        n_filaments = 0
        files = glob.glob(f'{output_directory}/*__{target}_coords.star')
        for j, f in enumerate(files):
            data = starfile.read(f)
            for filament_id in data['aisFilamentID'].unique():
                n_filaments += 1
                mask = data['aisFilamentID'] == filament_id
                filament_df = data.loc[mask].copy()  
                filament_df.loc[:, 'aisFilamentID'] = n_filaments 
                out_path = f.replace('.star', f'_filament_{int(filament_id)}_coords.star')
                starfile.write({"particles": filament_df}, out_path)
            os.remove(f)

    if filament:
        if per_filament_star_file:
            print(f"\n\033[38;5;208m{''}found {n_particles} particles along {n_filaments} filaments. that's {n_particles * spacing / 10000.0:.2f} um of {target} :) {''}\033[0m\n")
        else:
            print(f"\n\033[38;5;208m{''}found {n_particles} particles in total. that's {n_particles * spacing / 10000.0} um of {target} :) {''}\033[0m\n")
    else:
        print(f"\n\033[38;5;208m{''}found {n_particles} particles in total. {''}\033[0m\n")
    print(f"\033[33m"
          f"as a reminder, the WarpTools coordinate ingestion command is something like:\n\n"
          f"WarpTools ts_export_particles "
          f"--settings warp_tiltseries.settings "
          f"--input_directory {output_directory} "
          f"--coords_angpix 10.0 "
          f"--output_star relion/{target}/particles.star "
          f"--output_angpix 5.0 "
          f"--box 64 "
          f"--diameter 250 "
          f"--relative_output_paths "
          f"--3d "
          f"\n\n"
          f"(but make sure you adapt the parameters to your use case)\n"
          f"\033[0m")


def _load_2d_model(scnm_path):
    from keras.models import load_model as keras_load_model, clone_model
    from keras.layers import Input

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(scnm_path, 'r') as archive:
            archive.extractall(path=tmp)

        files = os.listdir(tmp)
        weights_file = next(f for f in files if f.endswith('_weights.h5'))
        metadata_file = next(f for f in files if f.endswith('_metadata.json'))

        with open(os.path.join(tmp, metadata_file)) as f:
            metadata = json.load(f)

        model = keras_load_model(os.path.join(tmp, weights_file), compile=False)
        new_input = Input(shape=(None, None, 1))
        new_model = clone_model(model, input_tensors=new_input)
        new_model.set_weights(model.get_weights())

    return new_model, metadata['title'], metadata['apix']


def _bin_volume(vol, b=1):
    if b == 1:
        return vol
    j, k, l = vol.shape
    vol = vol[:j // b * b, :k // b * b, :l // b * b]
    vol = vol.reshape((j // b, b, k // b, b, l // b, b)).mean(5).mean(3).mean(1)
    return vol


def _segment_tomo_2d(tomo_path, model, tta=1, model_apix=None, input_apix=None):
    with mrcfile.open(tomo_path) as m:
        volume = np.array(m.data, dtype=np.float32)
        volume_apix = float(m.voxel_size.x)

    if volume_apix <= 1.0 and input_apix is None:
        print(f'warning: {tomo_path} header lists voxel size as 1.0 A/px, assuming 10.0 Å/px.')
        volume_apix = 10.0

    original_shape = volume.shape
    scale = 1.0
    if model_apix is not None:
        apix_to_use = input_apix if input_apix is not None else volume_apix
        scale = apix_to_use / model_apix
        if abs(scale - 1.0) > 0.05:
            from scipy.ndimage import zoom
            volume = zoom(volume, scale, order=1)

    volume -= np.mean(volume)
    volume /= np.std(volume) + 1e-7

    segmented_volume = np.zeros_like(volume)

    if tta == 1:
        w = 32 * (volume.shape[1] // 32)
        w_pad = (volume.shape[1] % 32) // 2
        h = 32 * (volume.shape[2] // 32)
        h_pad = (volume.shape[2] % 32) // 2
        for j in range(volume.shape[0]):
            sl = volume[j, w_pad:w_pad + w, h_pad:h_pad + h][np.newaxis, :, :, np.newaxis]
            segmented_volume[j, w_pad:w_pad + w, h_pad:h_pad + h] = np.squeeze(model.predict(sl, verbose=0))
    else:
        r = [0, 1, 2, 3, 0, 1, 2, 3]
        f = [0, 0, 0, 0, 1, 1, 1, 1]
        for k in range(tta):
            volume_instance = np.rot90(volume, k=r[k], axes=(1, 2))
            segmented_instance = np.zeros_like(volume_instance)
            w = 32 * (volume_instance.shape[1] // 32)
            w_pad = (volume_instance.shape[1] % 32) // 2
            h = 32 * (volume_instance.shape[2] // 32)
            h_pad = (volume_instance.shape[2] % 32) // 2

            if f[k]:
                volume_instance = np.flip(volume_instance, axis=2)

            for j in range(volume_instance.shape[0]):
                sl = volume_instance[j, w_pad:w_pad + w, h_pad:h_pad + h][np.newaxis, :, :, np.newaxis]
                segmented_instance[j, w_pad:w_pad + w, h_pad:h_pad + h] = np.squeeze(model.predict(sl, verbose=0))

            if f[k]:
                segmented_instance = np.flip(segmented_instance, axis=2)
            segmented_instance = np.rot90(segmented_instance, k=-r[k], axes=(1, 2))
            segmented_volume += segmented_instance
        segmented_volume /= tta

    segmented_volume = np.clip(segmented_volume, 0.0, 1.0)

    # rescale output back to original shape
    if abs(scale - 1.0) > 0.05:
        from scipy.ndimage import zoom as _zoom
        sj, sk, sl = segmented_volume.shape
        oj, ok, ol = original_shape
        segmented_volume = _zoom(segmented_volume, (oj / sj, ok / sk, ol / sl), order=1)

    return segmented_volume, volume_apix


def _postprocess_2d(segmented_volume, model_apix):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(segmented_volume.astype(np.float32), sigma=model_apix, axis=0)


def _segmentation_thread_2d(tomogram_list, scnm_path, model_title, output_dir, gpu, tta, overwrite, model_apix, data_format='int8', input_apix=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    process_start_time = psutil.Process().create_time()

    print(f'GPU {gpu} - loading 2D model ({scnm_path}).')
    model, _, _ = _load_2d_model(scnm_path)

    for j, tomo_path in enumerate(tomogram_list, 1):
        tomo_name = os.path.splitext(os.path.basename(tomo_path))[0]
        output_file = os.path.join(output_dir, f"{tomo_name}__{model_title}.mrc")
        wrote_temporary = False
        try:
            if os.path.exists(output_file):
                temp_file_write_time = os.path.getmtime(output_file)
                if not overwrite or temp_file_write_time > process_start_time:
                    print(f"{j}/{len(tomogram_list)} (GPU {gpu}) - {model_title} - {os.path.basename(tomo_path)} (skipped)")
                    continue

            with mrcfile.new(output_file, overwrite=True) as m:
                m.set_data(np.zeros((10, 10, 10), dtype=np.float32))
                m.voxel_size = 1.0
                wrote_temporary = True

            segmented_volume, volume_apix = _segment_tomo_2d(tomo_path, model, tta=tta, model_apix=model_apix, input_apix=input_apix)
            segmented_volume = _postprocess_2d(segmented_volume, model_apix)
            segmented_volume = np.clip(segmented_volume, 0.0, 1.0)
            if data_format == 'float32':
                segmented_volume = segmented_volume.astype(np.float32)
            elif data_format == 'uint16':
                segmented_volume = (segmented_volume * 255).astype(np.uint16)
            else:  # int8 (default)
                segmented_volume = (segmented_volume * 127).astype(np.int8)

            with mrcfile.new(output_file, overwrite=True) as m:
                m.set_data(segmented_volume)
                m.voxel_size = volume_apix

            print(f"{j}/{len(tomogram_list)} (GPU {gpu}) - {model_title} - {os.path.basename(tomo_path)}")
        except Exception as e:
            if wrote_temporary and os.path.exists(output_file):
                os.remove(output_file)
            print(f"{j}/{len(tomogram_list)} (GPU {gpu}) - {model_title} - {os.path.basename(tomo_path)} - ERROR: {e}")


def dispatch_segment(feature, data_directory, output_directory, tta=1, batch_size=8, overwrite=False, data_format='int8', gpus=None, input_apix=None):
    import tensorflow as tf

    if isinstance(data_directory, (list, tuple)):
        patterns = list(data_directory)
    else:
        patterns = [data_directory]

    if output_directory is None:
        output_directory = 'segmented'

    if gpus is None:
        gpus = list(range(len(tf.config.list_physical_devices('GPU'))))
    else:
        gpus = [int(g) for g in gpus.split(',') if g.strip().isdigit()]

    if len(gpus) == 0:
        print("\033[93mwarning: no GPUs detected. processing will continue, but using CPUs only!\033[0m")
        gpus = [-1]

    tomograms = []
    for p in patterns:
        if os.path.isdir(p):
            tomograms.extend(glob.glob(os.path.join(p, '*.mrc')))
        else:
            tomograms.extend(glob.glob(p))
    tomograms = sorted(set(tomograms))

    print(f'easymode segment (2D)\n'
          f'feature: {feature}\n'
          f'data_patterns: {patterns}\n'
          f'output_directory: {output_directory}\n'
          f'gpus: {gpus}\n'
          f'tta: {tta}\n'
          f'overwrite: {overwrite}\n')
    print(f'Found {len(tomograms)} tomograms to segment.\n')

    if len(tomograms) == 0:
        return

    scnm_path, metadata = get_model(feature, _2d=True)
    if scnm_path is None:
        print(f'Could not find 2D model for {feature}! Exiting.')
        return

    model_apix = metadata['apix']
    model_title = metadata.get('title', feature)

    print(f'Using model: {scnm_path}, inference at {model_apix} Å/px.\n')

    os.makedirs(output_directory, exist_ok=True)

    multiprocessing.set_start_method('spawn', force=True)

    processes = []
    for gpu in gpus:
        p = multiprocessing.Process(
            target=_segmentation_thread_2d,
            args=(tomograms, scnm_path, model_title, output_directory, gpu, tta, overwrite, model_apix, data_format, input_apix)
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
        for path in glob.glob(os.path.join(output_directory, f'*__{model_title}.mrc')):
            if os.path.getsize(path) < 10_000:
                os.remove(path)
        return

    print()
    print('\033[92mSegmentation finished!\033[0m')
    print()
