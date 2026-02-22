import subprocess, glob, os, starfile
from easymode.core.distribution import get_model, load_model
from multiprocessing import cpu_count

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
                filament_df = data.loc[mask].copy()  # <- explicit copy avoids warning
                filament_df.loc[:, 'aisFilamentID'] = n_filaments  # <- safe assignment
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


def dispatch_segment(feature, data_directory, output_directory, tta=1, batch_size=8, overwrite=False, data_format='int8', gpus=None):
    import tensorflow as tf

    if output_directory is None:
        output_directory = 'segmented'

    if gpus is None:
        gpus = list(range(0, len(tf.config.list_physical_devices('GPU'))))
    else:
        gpus = [int(g) for g in gpus.split(',') if g.strip().isdigit()]
    gpus = ','.join([str(g) for g in gpus])

    if len(gpus) == 0:
        print("\033[93m" + "warning: no GPUs detected. processing will continue, but using CPUs only!" + "\033[0m")

    print(f'easymode segment\n'
          f'model_path: {feature}\n'
          f'input_data: {data_directory}\n'
          f'output_directory: {output_directory}\n'
          f'gpus: {gpus}\n'
          f'tta: {tta}\n'
          f'overwrite: {overwrite}\n'
          f'ais_2d_nets: True')

    patterns = data_directory if isinstance(data_directory, (list, tuple)) else [data_directory]

    # collect tomograms from dirs/files/patterns
    tomograms = []
    for p in patterns:
        if os.path.isdir(p):
            tomograms.extend(glob.glob(os.path.join(p, '*.mrc')))
        else:
            tomograms.extend(glob.glob(p))
    tomograms = sorted(set(tomograms))

    print(f'Found {len(tomograms)} tomograms to segment.\n')

    if len(tomograms) == 0:
        return

    model_path, metadata = get_model(feature, _2d=True)
    if model_path is None:
        print(f'Could not find model for {feature}! Exiting.')
        exit()

    model_apix = metadata['apix']

    data_arg = " ".join(patterns)
    command = f'ais segment -m {model_path} -apix {model_apix} -d {data_arg} -ou {output_directory} -tta {tta} -p 1 --overwrite {"1" if overwrite else "0"} -gpu {gpus}'

    _run(command)

