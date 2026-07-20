import gc
import glob
import multiprocessing
import os
import time
import traceback

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import mrcfile
import numpy as np
import psutil
import tensorflow as tf

from easymode.core.distribution import get_model, load_model
from easymode.n2n.streaming_legacy import (
    iter_legacy_tile_chunks,
    legacy_tile_count,
    place_legacy_predictions,
)


tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

TILE_SIZE = 160
OVERLAP = 32
DEFAULT_CHUNK_SIZE = 16


def _predict_adaptively(model, tiles, positions, batch_size):
    """Yield predictions in order, splitting a VRAM-limited chunk as needed."""

    try:
        prediction = np.asarray(
            model.predict(tiles, verbose=0, batch_size=batch_size)
        )
    except tf.errors.ResourceExhaustedError:
        number_of_tiles = int(tiles.shape[0])
        if number_of_tiles <= 1:
            raise

        fallback_chunk_size = max(1, number_of_tiles // 4)
        print(
            f"Memory error with streaming chunk size {number_of_tiles}, "
            f"retrying in chunks of at most {fallback_chunk_size}",
            flush=True,
        )
        gc.collect()

        for start in range(0, number_of_tiles, fallback_chunk_size):
            end = min(start + fallback_chunk_size, number_of_tiles)
            yield from _predict_adaptively(
                model,
                tiles[start:end],
                positions[start:end],
                batch_size,
            )
        return

    if prediction.shape[0] != tiles.shape[0]:
        raise RuntimeError(
            f"model returned {prediction.shape[0]} predictions for "
            f"{tiles.shape[0]} input tiles"
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("model returned NaN or infinite values")

    yield prediction, positions


def _denoise_tomogram_instance(
    volume,
    model,
    batch_size,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    """Denoise one transformed volume with exact legacy hard-core assembly.

    Tile extraction, zero padding, traversal order, 96^3 centre selection, and
    edge truncation are identical to upstream Easymode.  The only change is
    that input tiles and predictions are held in a bounded chunk and written to
    the output immediately instead of retaining every tile for the full volume.
    """

    original_shape = tuple(int(value) for value in volume.shape)
    number_of_tiles = legacy_tile_count(
        original_shape,
        patch_size=TILE_SIZE,
        overlap=OVERLAP,
    )

    print(
        "Streaming-legacy inference: "
        f"{number_of_tiles} tiles; "
        f"tile=({TILE_SIZE}, {TILE_SIZE}, {TILE_SIZE}); "
        f"core=({TILE_SIZE - 2 * OVERLAP}, "
        f"{TILE_SIZE - 2 * OVERLAP}, "
        f"{TILE_SIZE - 2 * OVERLAP}); "
        f"stride=({TILE_SIZE - 2 * OVERLAP}, "
        f"{TILE_SIZE - 2 * OVERLAP}, "
        f"{TILE_SIZE - 2 * OVERLAP}); "
        f"chunk<={chunk_size}.",
        flush=True,
    )

    denoised_volume = np.zeros(original_shape, dtype=np.float32)
    processed = 0
    progress_step = max(1, number_of_tiles // 20)
    next_progress = progress_step

    for tile_chunk, positions in iter_legacy_tile_chunks(
        volume,
        patch_size=TILE_SIZE,
        overlap=OVERLAP,
        chunk_size=chunk_size,
    ):
        for prediction, prediction_positions in _predict_adaptively(
            model,
            tile_chunk,
            positions,
            batch_size,
        ):
            place_legacy_predictions(
                denoised_volume,
                prediction,
                prediction_positions,
                patch_size=TILE_SIZE,
                overlap=OVERLAP,
            )
            processed += len(prediction_positions)

            if processed >= next_progress or processed == number_of_tiles:
                print(
                    f"Streaming-legacy progress: "
                    f"{processed}/{number_of_tiles} tiles",
                    flush=True,
                )
                while next_progress <= processed:
                    next_progress += progress_step

    if processed != number_of_tiles:
        raise RuntimeError(
            f"processed {processed} tiles, expected {number_of_tiles}"
        )

    tf.keras.backend.clear_session()
    gc.collect()
    return denoised_volume


def denoise_tomogram(
    model,
    tomogram_path,
    tta=1,
    batch_size=2,
    iter=1,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    if not 1 <= int(tta) <= 16:
        raise ValueError(f"tta must be between 1 and 16, received {tta}")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, received {batch_size}")
    if int(iter) <= 0:
        raise ValueError(f"iter must be positive, received {iter}")
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, received {chunk_size}")

    with mrcfile.open(tomogram_path) as m:
        volume = m.data.astype(np.float32)
        volume_apix = float(m.voxel_size.x)
    volume = np.pad(volume, pad_width=16, mode='reflect')

    # All 16 combinations of right-angle rotations and flips that respect the
    # anisotropy of the data.
    k_xy = [0, 2, 2, 0, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 3]
    k_fx = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    k_yz = [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]

    for _ in range(iter):
        volume -= np.mean(volume)
        volume /= np.std(volume) + 1e-6
        denoised_volume = np.zeros_like(volume)

        for j in range(tta):
            tta_vol = volume.copy()
            tta_vol = np.rot90(tta_vol, k=k_xy[j], axes=(1, 2))
            tta_vol = tta_vol if not k_fx[j] else np.flip(tta_vol, axis=1)
            tta_vol = np.rot90(tta_vol, k=2 * k_yz[j], axes=(0, 1))

            denoised_tta_vol = _denoise_tomogram_instance(
                tta_vol,
                model,
                batch_size,
                chunk_size=chunk_size,
            )

            denoised_tta_vol = np.rot90(
                denoised_tta_vol,
                k=-2 * k_yz[j],
                axes=(0, 1),
            )
            denoised_tta_vol = (
                denoised_tta_vol
                if not k_fx[j]
                else np.flip(denoised_tta_vol, axis=1)
            )
            denoised_tta_vol = np.rot90(
                denoised_tta_vol,
                k=-k_xy[j],
                axes=(1, 2),
            )
            denoised_volume += denoised_tta_vol

            del tta_vol
            del denoised_tta_vol
            gc.collect()

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
    'n2n': 'n2n_direct',
    'ddw': 'ddw_direct',
    'iso': 'iso_direct',
}


def _remove_temporary_markers(output_dir):
    """Remove only Easymode's 10^3 all-minus-one failure placeholders."""

    removed = []
    for path in glob.glob(os.path.join(output_dir, '*.mrc')):
        try:
            if os.path.getsize(path) > 10_000:
                continue
            with mrcfile.open(path, permissive=True) as m:
                is_marker = (
                    m.data.shape == (10, 10, 10)
                    and m.data.dtype == np.float32
                    and np.all(m.data == -1.0)
                )
            if is_marker:
                os.remove(path)
                removed.append(path)
        except Exception:
            continue
    return removed


def denoiser_thread(
    tomogram_list,
    model_path,
    output_dir,
    gpu,
    batch_size,
    tta,
    overwrite,
    iter,
    chunk_size,
):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)

    process_start_time = psutil.Process().create_time()

    print(f'GPU {gpu} - loading model ({model_path}).', flush=True)
    model = load_model(model_path)

    print(f'GPU {gpu} - starting inference.', flush=True)
    failures = []

    for j, tomo_path in enumerate(tomogram_list, start=1):
        tomo_name = os.path.splitext(os.path.basename(tomo_path))[0]
        output_file = os.path.join(output_dir, f'{tomo_name}.mrc')
        wrote_temporary = False

        try:
            if os.path.exists(output_file):
                file_age = os.path.getmtime(output_file)
                if not overwrite or file_age > process_start_time - 60:
                    continue

            with mrcfile.new(output_file, overwrite=True) as m:
                m.set_data(-1.0 * np.ones((10, 10, 10), dtype=np.float32))
                wrote_temporary = True

            denoised_volume, volume_apix = denoise_tomogram(
                model,
                tomo_path,
                tta,
                batch_size,
                iter=iter,
                chunk_size=chunk_size,
            )
            save_mrc(
                denoised_volume,
                output_file,
                data_format='float32',
                voxel_size=volume_apix,
            )

            etc = time.strftime(
                '%H:%M:%S',
                time.gmtime(
                    (time.time() - process_start_time)
                    / j
                    * (len(tomogram_list) - j)
                ),
            )
            print(
                f"{j}/{len(tomogram_list)} (on GPU {gpu}) - "
                f"{os.path.basename(output_file)} - etc: {etc}",
                flush=True,
            )
        except Exception as error:
            failures.append((tomo_path, str(error)))
            if wrote_temporary and os.path.exists(output_file):
                os.remove(output_file)
            print(
                f"{j}/{len(tomogram_list)} (on GPU {gpu}) - "
                f"{os.path.basename(output_file)} - ERROR: {error}",
                flush=True,
            )
            traceback.print_exc()

    if failures:
        raise RuntimeError(
            f"{len(failures)} tomogram(s) failed during denoising on GPU {gpu}"
        )


def dispatch(
    input_directory,
    output_directory,
    method='n2n',
    tta=1,
    batch_size=8,
    overwrite=False,
    iter=1,
    gpus='0',
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    if os.path.abspath(output_directory) == os.path.abspath(input_directory):
        raise ValueError(
            'Please choose an output directory that is different from the '
            'input directory; original volumes must not be overwritten.'
        )

    if method not in METHOD_TO_WEIGHTS:
        raise ValueError(
            f"unknown denoising method {method!r}; "
            f"available: {sorted(METHOD_TO_WEIGHTS)}"
        )
    if not 1 <= int(tta) <= 16:
        raise ValueError(f"tta must be between 1 and 16, received {tta}")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, received {batch_size}")
    if int(iter) <= 0:
        raise ValueError(f"iter must be positive, received {iter}")
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, received {chunk_size}")

    if gpus is None:
        gpus = list(range(len(tf.config.list_physical_devices('GPU'))))
    else:
        gpus = [int(g) for g in gpus.split(',') if g.strip().isdigit()]

    if len(gpus) == 0:
        print(
            '\033[93mwarning: no GPUs detected. Processing will continue '
            'using CPUs only.\033[0m'
        )
        gpus = [-1]

    print(
        f'easymode denoise\n'
        f'method: {method}\n'
        f'data_directory: {input_directory}\n'
        f'output_directory: {output_directory}\n'
        f'inference_mode: streaming-legacy\n'
        f'gpus: {gpus}\n'
        f'tta: {tta}\n'
        f'overwrite: {overwrite}\n'
        f'batch_size: {batch_size}\n'
        f'chunk_size: {chunk_size}\n'
    )

    tomograms = sorted(glob.glob(os.path.join(input_directory, '*.mrc')))
    print(f'Found {len(tomograms)} tomograms to denoise in {input_directory}.')

    model_path = get_model(METHOD_TO_WEIGHTS[method])[0]
    if model_path is None:
        raise RuntimeError(
            f"could not locate or download weights for {METHOD_TO_WEIGHTS[method]}"
        )

    os.makedirs(output_directory, exist_ok=True)
    multiprocessing.set_start_method('spawn', force=True)

    processes = []
    for gpu in gpus:
        process = multiprocessing.Process(
            target=denoiser_thread,
            args=(
                tomograms,
                model_path,
                output_directory,
                gpu,
                batch_size,
                tta,
                overwrite,
                iter,
                chunk_size,
            ),
        )
        processes.append(process)
        process.start()
        time.sleep(2)

    for process in processes:
        process.join()

    failed_processes = [
        process
        for process in processes
        if process.exitcode not in (0, None)
    ]
    if failed_processes:
        removed = _remove_temporary_markers(output_directory)
        for path in removed:
            print(f'Removed temporary failure marker: {path}', flush=True)

        details = []
        for process in failed_processes:
            if process.exitcode is not None and process.exitcode < 0:
                details.append(
                    f'pid={process.pid}, signal={-process.exitcode}'
                )
            else:
                details.append(
                    f'pid={process.pid}, exitcode={process.exitcode}'
                )
        raise RuntimeError(
            'denoising worker failure(s): ' + ', '.join(details)
        )
