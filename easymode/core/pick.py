import subprocess, glob, os, starfile

def _run(cmd, capture=False):
    print(f'\033[42m{cmd}\033[0m\n')
    ret = subprocess.run(cmd, shell=True, capture_output=capture, text=True if capture else None)
    if ret.returncode != 0:
        print(f'\033[91merror running {cmd}\033[0m')
        exit()
    return ret.stdout


def pick(data_directory, target, output_directory, spacing, size, binning=2, processes=112, tomostar=True):
    print(f'easymode pick\n'
          f'feature: {target}\n'
          f'data_directory: {data_directory}\n'
          f'output_directory: {output_directory}\n'
          f'output_pattern: *__{target}_coords.star\n'
          f'spacing: {spacing} Å\n'
          f'size: {size} Å^3\n'
          f'binning: {binning}\n'
          f'n_processes: {processes}\n'
          f'rename to .tomostar: {tomostar}\n')

    command = f'ais pick -t {target} -d {data_directory} -ou {output_directory} -spacing {spacing} -size {size} -b {binning} -p {processes}'
    _run(command)
    if tomostar:
        files = glob.glob(f'{output_directory}/*__{target}_coords.star')
        n_particles = 0
        for j, f in enumerate(files):
            data = starfile.read(f)
            n_particles += len(data)
            tomo = os.path.basename(f.split('_10.00Apx')[0]) + '.tomostar'
            data["rlnMicrographName"] = tomo
            starfile.write({"particles": data}, f)

    print(f"\n\033[38;5;208m{''}found {n_particles} particles in total. {''}\033[0m\n")
    print(f"\033[33m"
          f"as a reminder, the WarpTools coordinate ingestion command is something like:\n\n"
          f"WarpTools ts_export_particles "
          f"--settings warp_tiltseries.settings "
          f"--input_directory {output_directory} "
          f"--coords_angpix 10.0 "
          f"--output_star relion/{output_directory.strip().split('/')[-1]}/particles.star "
          f"--output_angpix 5.0 "
          f"--box 64 "
          f"--diameter 250 "
          f"--relative_output_paths "
          f"--2d "
          f"\n\n"
          f"(but make sure you adapt the parameters to your use case)\n"
          f"\033[0m")