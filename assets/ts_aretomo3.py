#!/usr/bin/env python3
"""
ts_aretomo3.py -- minimal AreTomo3 wrapper for Warp tilt-series alignment.

Aligns every tilt stack under warp_tiltseries/tiltstack/*/ and copies the
resulting *_Imod/ folders into warp_tiltseries/alignments/.

After this script, run:
    WarpTools ts_import_alignments \\
        --settings warp_tiltseries.settings \\
        --alignments warp_tiltseries/alignments \\
        --alignment_angpix <acquisition pixel size>

Usage:
    python ts_aretomo3.py --aretomo3 /path/to/AreTomo3 --gpus 0,1,2,3
"""
import argparse, glob, multiprocessing, os, shutil, subprocess, time

ARETOMO3_FLAGS = (
    "-CorrCTF 0 -TiltCor 1 -Cmd 1 -Serial 1 "
    "-VolZ 0 -AtBin 8 -AlignZ 0 -SplitSum 0 -OutImod 1"
)


def align_worker(aretomo3, tilt_dirs, gpu):
    for d in tilt_dirs:
        name = os.path.basename(d)
        if glob.glob(os.path.join(d, '*_Imod')):
            print(f'[gpu {gpu}] {name}: already aligned, skipping')
            continue
        cmd = (
            f'{aretomo3} -InPrefix {d}/ -InSuffix .st -OutDir {d}/ '
            f'{ARETOMO3_FLAGS} -Gpu {gpu}'
        )
        print(f'[gpu {gpu}] {name}: aligning')
        ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if ret.returncode != 0:
            print(f'[gpu {gpu}] {name}: FAILED\n{ret.stderr}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--aretomo3', required=True, help='Path to the AreTomo3 executable.')
    p.add_argument('--gpus', default='0', help='Comma-separated GPU ids (default: 0).')
    p.add_argument('--root', default='warp_tiltseries',
                   help='Warp tilt-series processing folder (default: warp_tiltseries).')
    args = p.parse_args()

    tilt_dirs = sorted(
        d for d in glob.glob(os.path.join(args.root, 'tiltstack', '*')) if os.path.isdir(d)
    )
    if not tilt_dirs:
        raise SystemExit(f'no tilt-stack directories under {args.root}/tiltstack/')

    gpus = [g.strip() for g in args.gpus.split(',') if g.strip()]
    shares = [tilt_dirs[i::len(gpus)] for i in range(len(gpus))]

    t0 = time.time()
    procs = [
        multiprocessing.Process(target=align_worker, args=(args.aretomo3, share, gpu))
        for gpu, share in zip(gpus, shares)
    ]
    for proc in procs: proc.start()
    for proc in procs: proc.join()
    print(f'\nAreTomo3 finished in {time.time() - t0:.0f} s ({len(tilt_dirs)} tilt series).\n')

    alignments_dir = os.path.join(args.root, 'alignments')
    os.makedirs(alignments_dir, exist_ok=True)
    for imod in glob.glob(os.path.join(args.root, 'tiltstack', '*', '*_Imod')):
        name = os.path.basename(imod).split('_Imod')[0]
        shutil.copytree(imod, os.path.join(alignments_dir, name), dirs_exist_ok=True)
    print(f'alignment files copied to {alignments_dir}/')
    print(f'next step:\n    WarpTools ts_import_alignments --settings warp_tiltseries.settings '
          f'--alignments {alignments_dir} --alignment_angpix <acquisition pixel size>')


if __name__ == '__main__':
    main()
