import os, time, glob, multiprocessing, subprocess, shutil, json, mrcfile, tifffile
import easymode.core.config as cfg

def _run(cmd, capture=False, ignore_error=False):
    print(f'\033[42m{cmd}\033[0m\n')
    ret = subprocess.run(cmd, shell=True, capture_output=capture, text=True if capture else None)
    if ret.returncode != 0:
        print(f'\033[91merror running {cmd}\033[0m')
        if not ignore_error:
            exit()
        print(f'\033[93mcontinuing despite error...\033[0m')
    return ret.stdout

def _aretomo3_thread(tomo_list, gpu):
    t_start = time.time()

    print(f'{cfg.settings["ARETOMO3_ENV"]}')
    for j, tomo in enumerate(tomo_list):
        tomo_dir = os.path.join('warp_tiltseries', 'tiltstack', os.path.splitext(os.path.basename(tomo))[0])
        lock_path = os.path.join(tomo, '.lock')
        if not os.path.exists(lock_path):
            with open(lock_path, 'w') as f:
                f.write('')
            subprocess.run(f'{cfg.settings["ARETOMO3_PATH"]} -InPrefix {tomo_dir}/ -InSuffix .st -OutDir {tomo_dir}/ -CorrCTF 0 -TiltCor 1 -Cmd 1 -Serial 1 -VolZ 0 -AtBin 8 -AlignZ 0 -SplitSum 0 -OutImod 1 -Gpu {gpu}', shell=True, check=True, stdout=subprocess.DEVNULL)
            etc = (time.time() - t_start) / (j + 1) * (len(tomo_list) - j)
            etc_h = int(etc // 3600)
            etc_m = int((etc % 3600) // 60)
            etc_s = int(etc % 60)
            print(f'{j + 1}/{len(tomo_list)} (GPU {gpu}) - done aligning {tomo} - estimated time to completion: {etc_h}h {etc_m}m {etc_s}s')

    print(f'GPU {gpu} thread finished in {time.time() - t_start:.2f} seconds.')

def _aretomo_dispatch(tomo_list):
    t_start = time.time()
    tomo_dir = os.path.join('warp_tiltseries', 'tiltstack', os.path.splitext(os.path.basename(tomo_list[0]))[0])
    print(f'Base command: \033[42m{cfg.settings["ARETOMO3_PATH"]} -InPrefix {tomo_dir}/ -InSuffix .st -OutDir {tomo_dir}/ -CorrCTF 0 -TiltCor 1 -Cmd 1 -FlipVol 1 -Serial 1 -VolZ 0 -AtBin 8 -AlignZ 0 -SplitSum 0 -OutImod 1\033[0m\n')

    for t in tomo_list:
        if os.path.exists(os.path.join(t, '.lock')):
            os.remove(os.path.join(t, '.lock'))

    processes = list()
    PER_DEVICE = 1
    gpus = get_gpu_list()
    for gpu in gpus:
        for i in range(PER_DEVICE):
            p = multiprocessing.Process(target=_aretomo3_thread, args=(tomo_list, gpu))
            print(f'Launching AreTomo3 on GPU ID {gpu} (thread {i}).')
            processes.append(p)
            p.start()
            time.sleep(0.1)

    for p in processes:
        p.join()

    print(f'\033[38;5;208mAreTomo alignment completed in {time.time() - t_start:.2f} seconds ({(time.time() - t_start) / len(tomo_list):.2f} per tilt series).\033[0m')

def find_extension(frames_path):
    extensions = {
        '*.eer': 0,
        '*.mrc': 0,
        '*.tif': 0,
        '*.tiff': 0,
    }
    for e in extensions:
        extensions[e] = len(glob.glob(os.path.join(frames_path, e)))
    return max(extensions, key=extensions.get)

def find_shape(frames_path, extension):
    if extension == '.eer':
        return 4096, 4096
    elif extension == '.mrc':
        import mrcfile
        sample_file = glob.glob(os.path.join(frames_path, f'*{extension}'))[0]
        with mrcfile.open(sample_file, permissive=True) as mrc:
            return mrc.data.shape[-2], mrc.data.shape[-1]
    elif extension in ['.tif', '.tiff']:
        import tifffile
        sample_file = glob.glob(os.path.join(frames_path, f'*{extension}'))[0]
        with tifffile.TiffFile(sample_file) as tif:
            return tif.pages[0].shape[-2], tif.pages[0].shape[-1]
    return 4000, 4000

def get_gpu_list():
    try:
        # Try NVIDIA first
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True)
        gpu_indices = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
        return gpu_indices
    except:
        # Fallback: try to detect via /proc (Linux) or other methods
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True)
            gpu_lines = [line for line in result.stdout.split('\n') if 'VGA' in line or 'Display' in line]
            return list(range(len(gpu_lines))) if gpu_lines else []
        except:
            return []



def reconstruct(frames, mdocs, apix, dose, extension=None, tomo_apix=10.0, thickness=3000, shape=None, steps='1111111', halfmaps=True):
    root = os.getcwd()
    frames_path = frames if os.path.exists(frames) else os.path.join(root, frames)
    mdoc_path = mdocs if os.path.exists(mdocs) else os.path.join(root, mdocs)
    extension = extension if extension is not None else find_extension(frames_path)
    extension = f'.{extension}' if not '.' in extension else extension

    print(f'Easymode warp reconstruct - settings:'
          f'\n  root: {root}'
          f'\n  frames path: {frames_path}'
          f'\n  mdoc path: {mdoc_path}'
          f'\n  extension: {extension}'
          f'\n  apix: {apix}'
          f'\n  dose: {dose}'
          f'\n  tomo apix: {tomo_apix}'
          f'\n  thickness: {thickness}'
          f'\n  shape: {shape if shape is not None else "auto"}'
          f'\n  steps: {steps}')

    print(f'\n\033[96mRunning Warp easymode in {root}.\033[0m')

    print(f'\n\033[96mCreating settings (frame series)\033[0m')
    _run(f'WarpTools create_settings --folder_data {frames_path} --folder_processing warp_frameseries --output warp_frameseries.settings --extension "*{extension}" --angpix {apix} --exposure {dose}')

    print(f'\n\033[96mCreating settings (tilt series)\033[0m')
    tomo_size = [int(f) for f in shape.split('x')] if shape is not None else find_shape(frames_path, extension)
    _run(f'WarpTools create_settings --folder_data tomostar --output warp_tiltseries.settings --folder_processing warp_tiltseries --extension "*.tomostar" --angpix {apix} --exposure {dose} --tomo_dimensions {tomo_size[0]}x{tomo_size[1]}x{int(thickness // apix)}')

    steps = [s == '1' for s in steps]

    if steps[0]:
        print(f'\n\033[96mMotion correction & CTF estimation\033[0m')
        _run(f'WarpTools fs_motion_and_ctf --settings warp_frameseries.settings --c_grid 2x2x1 --c_defocus_max 8 --c_use_sum --out_averages {"--out_average_halves" if halfmaps else ""} --perdevice {"1" if extension == "*.eer" else "2"} --c_range_max {2 * apix}')

    if steps[1]:
        print(f'\n\033[96mImporting tiltseries\033[0m')
        _run(f'WarpTools ts_import --mdocs {mdoc_path} --frameseries warp_frameseries --tilt_exposure {dose} --min_intensity 0.3 --dont_invert --output tomostar --override_axis')

    if steps[2]:
        print(f'\n\033[96mAssembling .st files\033[0m')
        _run(f'WarpTools ts_stack --settings warp_tiltseries.settings --perdevice 1')

    # auto AreTomo
    if steps[3]:
        print(f'\n\033[96mAligning with AreTomo (/public/EM/AreTomo/Aretomo)\033[0m')
        tomos = sorted([f for f in glob.glob(os.path.join(root, 'warp_tiltseries', 'tiltstack', '*')) if os.path.isdir(f)])
        _aretomo_dispatch(tomos)

    if steps[4]:
        print(f'\n\033[96mOrganising alignment files\033[0m')
        os.makedirs(os.path.join(root, 'warp_tiltseries', 'alignments'), exist_ok=True)
        dirs = glob.glob(os.path.join(root, 'warp_tiltseries', 'tiltstack', '*', '*Imod'))
        for d in dirs:
            shutil.copytree(d, os.path.join(root, 'warp_tiltseries', 'alignments', os.path.basename(d).split('_Imod')[0]), dirs_exist_ok=True)
        print(f".xf, .aln, etc. are in {os.path.join(root, 'warp_tiltseries', 'alignments')}")

        print(f'\n\033[96mParsing alignments in /warp_tiltseries/alignments/\033[0m')
        _run(f"WarpTools ts_import_alignments --settings warp_tiltseries.settings --alignments warp_tiltseries/alignments --alignment_angpix {apix}")

    if steps[5]:
        print(f'\n\033[96mEstimating tilt series CTF\033[0m')
        _run(f'WarpTools ts_ctf --settings warp_tiltseries.settings --range_high 10.0 --defocus_max 8 --perdevice 1')

    if steps[6]:
        print(f'\n\033[96mChecking handedness\033[0m')
        std_out_hand = _run(f'WarpTools ts_defocus_hand --settings warp_tiltseries.settings --check', capture=True, ignore_error=True)
        print(std_out_hand)
        correlation = 1.0
        for line in std_out_hand.split('\n'):
            if 'Average correlation:' in line:
                correlation = float(line.split('Average correlation: ')[1].strip())
                break
        if correlation < 0.0:
            print(f'\033[38;5;208mCorrecting handedness!.\n\033[0m')
            _run(f'WarpTools ts_defocus_hand --settings warp_tiltseries.settings --set_flip')

        print(f'\n\033[96mReconstructing volumes\033[0m')
        _run(f'WarpTools ts_reconstruct --settings warp_tiltseries.settings --angpix {tomo_apix} --dont_invert {"--halfmap_frames" if halfmaps else ""} --perdevice 1')

        n_mdocs = len(glob.glob(os.path.join(mdoc_path, '*.mdoc')))
        n_tomo_out = len(glob.glob(os.path.join(root, 'warp_tiltseries', 'reconstruction', f'*{10.00:.2f}Apx.mrc')))
        print(f'\033[38;5;208m{n_tomo_out} succesfully reconstructed, {n_mdocs - n_tomo_out} failed.\n\033[0m')
