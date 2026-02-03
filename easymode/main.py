import argparse
import easymode.core.config as cfg
import os

# TODO: clear cache command

def main():
    parser = argparse.ArgumentParser(description="easymode: pretrained general networks for cellular cryoET.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    if os.path.exists('/lmb/home/mlast/easymode_dev'):
        train_parser = subparsers.add_parser('train', description='Train an easymode network.')
        train_parser.add_argument('-t', "--title", type=str, required=True, help="Title of the model.")
        train_parser.add_argument('-f', "--features", nargs="+", required=True, help="List of features to train on, e.g. 'Ribosome3D Junk3D' - corresponding data directories are expected in /cephfs/mlast/compu_projects/easymode/training/3d/data/{features}")
        train_parser.add_argument('-e', "--epochs", type=int, help="Number of epochs to train for (default 200).", default=200)
        train_parser.add_argument('-b', "--batch_size", type=int, help="Batch size for training (default 8).", default=8)
        train_parser.add_argument('-ls', "--lr_start", type=float, help="Initial learning rate for the optimizer (default 5e-3).", default=5e-3)
        train_parser.add_argument('-le', "--lr_end", type=float, help="Final learning rate for the optimizer (default 5e-5).", default=5e-5)

    set_params = subparsers.add_parser('set', help='Set environment variables.')
    set_params.add_argument('--cache-directory', type=str, help="Path to the directory to store and search for easymode network weights in.")
    set_params.add_argument('--aretomo3-path', type=str, help="Path to the AreTomo3 executable.")
    set_params.add_argument('--aretomo3-env', type=str, help="Command to initialize the AreTomo3 environment, e.g. 'module load aretomo/3.1.0'")

    subparsers.add_parser('list', help='List the features for which pretrained general segmentation networks are available.')

    if os.path.exists('/lmb/home/mlast/easymode_dev'):
        package = subparsers.add_parser('package', description='Package model and weights. Note that this is used for 3D models only; 2D models are packaged and distributed with Ais.')
        package.add_argument('-c', "--checkpoint_directory", type=str, required=True, help="Path to the checkpoint directory to package from.")
        package.add_argument('-t', "--title", type=str, default=None, help="Title of the model to package. If not provided, the name of the checkpoint directory is used.")

    reconstruct = subparsers.add_parser('reconstruct', help='Reconstruct tomograms using WarpTools and AreTomo3.')
    reconstruct.add_argument('--frames', type=str, required=True, help="Directory containing raw frames.")
    reconstruct.add_argument('--mdocs', type=str, required=True, help="Directory containing mdocs.")
    reconstruct.add_argument('--apix', type=float, required=False, default=None, help="Pixel size of the frames in Angstrom. Leave empty to infer from mdoc.")
    reconstruct.add_argument('--dose', type=float, required=False, default=None, help="Dose per frame in e-/A^2. Leave empty to infer from mdoc.")
    reconstruct.add_argument('--extension', type=str, default=None, help="File extension of the frames (default: auto).")
    reconstruct.add_argument('--tomo_apix', type=float, default=10.0, help="Pixel size of the tomogram in Angstrom (default: 10.0). Easymode networks were all trained at 10.0 A/px.")
    reconstruct.add_argument('--thickness', type=float, default=3000.0, help="Thickness of the tomogram in Angstrom (default: 3000).")
    reconstruct.add_argument('--shape', type=str, default=None, help="Frame shape (e.g. 4096x4096). If not provided, the shape is inferred from the data.")
    reconstruct.add_argument('--steps', type=str, default='11111111', help="8-character string indicating which processing steps to perform (default: '1111111'). Each character corresponds to a specific step: 1 to perform the step, 0 to skip it. The steps are: 1) Frame motion and CTF, 2) Importing tilt series, 3) Creating tilt stacks, 4) Tilt series alignment, 5) Import alignments, 6) Tilt series CTF, 7) Check handedness, 8) Reconstruct volumes.")
    reconstruct.add_argument('--no_halfmaps', dest='halfmaps', action='store_false', help="If set, do not generate half-maps during motion correction or tomogram reconstruction. This precludes most methods of denoising.")
    reconstruct.add_argument('--force_align', action='store_true', help="If set, force AreTomo3 alignment of tilt series even if alignment files are already present.")

    segment = subparsers.add_parser('segment', help='Segment data using pretrained easymode networks.')
    segment.add_argument( "features", metavar='FEATURE', nargs="+", type=str, help="One or more features to segment (e.g. 'ribosome membrane microtubule'). Use 'easymode list' to see available features.")
    segment.add_argument("--data", nargs="+", type=str, required=True, help="One or more directories, file paths, or glob patterns. Examples: 'volumes', 'volumes/035*.mrc volumes/036*.mrc'.")
    segment.add_argument('--tta', required=False, type=int, default=4, help="Integer between 1 and 16. For values > 1, test-time augmentation is performed by averaging the predictions of several transformed versions of the input. Higher values can yield better results but increase computation time. (default: 4)")
    segment.add_argument('--output', required=False, type=str, default="segmented", help="Directory to save the output (default: ./segmented/)")
    segment.add_argument('--overwrite', action='store_true', help='If set, overwrite existing segmentations in the output directory.')
    segment.add_argument('--batch', type=int, default=1, help='Batch size for segmentation (default 1). Volumes are processed in batches of 160x160x160 shaped tiles. In/decrease batch size depending on available GPU memory.')
    segment.add_argument('--format', type=str, choices=['float32', 'uint16', 'int8'], default='int8', help='Output format for the segmented volumes (default: int8).')
    segment.add_argument('--gpu', type=str, default=None, help="Comma-separated list of GPU ids to use (leave empty to use all available devices).")
    segment.add_argument('--apix', type=float, default=None, help="Override the pixel size stored in the .mrc header (in Angstrom). Use this if the pixel size is missing or incorrect. Set to 0.0 to disallow any scaling.")
    segment.add_argument('--2d', dest="use_2d", action='store_true', help='Use the alternative Ais 2D/2.5D UNet instead of easymode 3D UNets.')

    report = subparsers.add_parser('report', help='Help improve easymode by reporting model failures and sharing the relevant volumes. All data will be kept confidential and is never released publicly.')
    report.add_argument("--tomogram", type=str, help="Path to the .mrc file to upload")
    report.add_argument("--model", type=str, help="Model that gave unsatisfactory results (e.g. 'ribosome')")
    report.add_argument('--contact', type=str, required=False, default="", help="Your email address (optional).")
    report.add_argument('--comment', type=str, required=False, default="", help="Additional comments (optional).")

    pick = subparsers.add_parser('pick', help='Pick particles in segmented volumes.')
    pick.add_argument("feature", metavar='FEATURE', type=str, help="Feature to pick, based on segmentations.")
    pick.add_argument('--data', required=True, type=str, help="Path to directory containing input .mrc's.")
    pick.add_argument('--output', required=True, type=str, default=None, help="Directory to save output coordinate files to.")
    pick.add_argument('--threshold', required=False, type=float, default=128, help="Threshold to apply to volumes prior to finding local maxima (default 128). Regardless of the segmentation .mrc dtype, the value range is assumed to be 0-255.")
    pick.add_argument('--binning', required=False, type=int, default=2, help="Binning factor to apply before processing (faster, possibly less accurate). Default is 2.")
    pick.add_argument('--spacing', required=False, type=float, default=10.0, help="Minimum distance between particles in Angstrom.")
    pick.add_argument('--size', required=False, type=float, default=10.0, help="Minimum particle size in cubic Angstrom.")
    pick.add_argument('--no_tomostar', dest='tomostar', action='store_false',help='Include this flag in order NOT to rename tomograms in the .star files from etc_10.00Apx.mrc to etc.tomostar.')
    pick.add_argument("--filament", action='store_true', help="Trace filaments & sample coordinates at a given spacing. If not set, will pick isolated particles. See Ais docs.")
    pick.add_argument('--length', required=False, type=float, default=500.0, help="For filament tracing: minimum filament length to place picks along, in Angstrom (default 500).")
    pick.add_argument('--separate_filaments', action='store_true', help='For filament tracing: write one .star file per filament instead of one per tomogram.')
    pick.add_argument('--centroid', action='store_true', help='When picking globular particles, set this flag to sample coordinates at the centroid of each connected component instead of at the deepest point in the threshold level isosurface. Use only when individual particles are well separated!')
    pick.add_argument('--min_particles', type=int, default=0, help="Minimum number of particles per tomogram to output a .star file (default 0). If fewer particles are found, no .star file is written for that tomogram.")

    denoise = subparsers.add_parser('denoise', help='Denoise or enhance contrast of tomograms.')
    denoise.add_argument('--data', type=str, required=True, help="Directory containing tomograms to denoise. In mode 'splits', this directory is expected to contain two subdirectories 'even' and 'odd' with the respective tomogram splits.")
    denoise.add_argument('--output', type=str, required=True, help="Directory to save denoised tomograms to.")
    denoise.add_argument('--mode', type=str, choices=['splits', 'direct'], help="Denoising mode. splits: statistically sound denoising of independent even/odd splits. direct: denoise the complete tomogram using a network that was trained on even/odd split denoised data. This last option helps avoid having to generate even/odd frame and volume splits.", default='direct')
    denoise.add_argument('--method', type=str, choices=['n2n', 'ddw'], help="Choose between denoising methods: 'n2n' for Noise2Noise (e.g. cryoCARE), or 'ddw' for DeepDeWedge. See github.com/juglab/cryocare_pip and github.com/mli-lab/deepdewedge and corresponding publications. In easymode we use custom tensorflow implementations.", default='n2n')
    denoise.add_argument('--tta', type=int, default=1, help="Test-time augmentation factor (default 1). Input volumes can be processed multiple times in different orientations and the results averaged to yield a (potentially) better result. Higher values increase computation time. Maximum is 16, default is 1.")
    denoise.add_argument('--overwrite', action='store_true', help='If set, overwrite existing segmentations in the output directory.')
    denoise.add_argument('--batch', type=int, default=1,help='Batch size for segmentation (default 1). Volumes are processed in batches of 128x128x128 shaped tiles. In/decrease batch size depending on available GPU memory.')
    denoise.add_argument('--iter', type=int, default=1, help="Only valid in direct mode: number of denoising iterations to perform (default 1). If you are really starved for contrast, try increasing this - but beware of artifacts.")
    denoise.add_argument('--gpu', type=str, default='0,', help="Comma-separated list of GPU ids to use (default '0').")

    select_tilts = subparsers.add_parser('select_tilts', help='Automatic marking of tilt images to be excluded from tomogram reconstruction')
    select_tilts.add_argument('--tiltstack', type=str, required=False, default="warp_tiltseries/tiltstack", help="Directory containing tilt stacks OR path to a single tilt stack OR glob pattern for multiple tilt stack files.")
    select_tilts.add_argument('--tomostar', type=str, required=False, default=None, help="Directory containing Warp-style tomogram star files.")
    select_tilts.add_argument('--xml', type=str, required=False, default=None, help="Directory containing Warp-style tilt series .xml files. If provided, xmls will be updated to reflect excluded tilts.")
    select_tilts.add_argument('--output', type=str, required=False, default=None, help="Directory to save output tomogram star files.")
    select_tilts.add_argument('--tta', type=int, required=False, default=1, help="Test-time augmentation factor (default 1). Input images can be processed multiple times in different orientations and the results averaged to yield a (potentially) better result. Higher values increase computation time. Maximum is 8, default is 1.")
    select_tilts.add_argument('--gpu', type=str, default='0,', help="Comma-separated list of GPU ids to use (default '0').")
    select_tilts.add_argument('--overwrite', action='store_true', help='If set, overwrite existing .tomostar files. When processing a directory of tomostar files, a backup of these files is always created regardless of this setting.')
    select_tilts.add_argument('--extension', type=str, default='*.tomostar', help='Filetype extension of the tomogram star files. Default: *.tomostar')
    select_tilts.add_argument('--threshold', type=float, default=0.5, help='Minimum score for a tilt to be kept. Network scores tilts 0.0 to 1.0, where 0 is bad and 1 is good. Default: 0.5')

    if os.path.exists('/lmb/home/mlast/easymode_dev'):
        tilt_train = subparsers.add_parser('tilt_train', description='Train tilt selection network.')
        tilt_train.add_argument('-e', "--epochs", type=int, help="Number of epochs to train for (default 200).", default=200)
        tilt_train.add_argument('-b', "--batch_size", type=int, help="Batch size for training (default 32).", default=32)
        tilt_train.add_argument('-ls', "--lr_start", type=float, help="Initial learning rate for the optimizer (default 5e-3).", default=1e-3)
        tilt_train.add_argument('-le', "--lr_end", type=float, help="Final learning rate for the optimizer (default 5e-5).", default=1e-5)


    args = parser.parse_args()

    if args.command == 'train':
        from easymode.segmentation.train import train_model
        train_model(title=args.title,
                    features=args.features,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr_start=args.lr_start,
                    lr_end=args.lr_end,
                    )
    elif args.command == 'tilt_train':
        from easymode.tiltfilter.train import train_model
        train_model(batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr_start=args.lr_start,
                    lr_end=args.lr_end, )
    elif args.command == 'denoise':
        if args.method == 'n2n':
            import easymode.n2n.inference as n2n
            n2n.dispatch(mode=args.mode,
                     input_directory=args.data,
                     output_directory=args.output,
                     tta=args.tta,
                     batch_size=args.batch,
                     overwrite=args.overwrite,
                     iter=args.iter,
                     gpus=args.gpu)
        elif args.method == 'ddw':
            import easymode.ddw.inference as ddw
            ddw.dispatch(mode=args.mode,
                     input_directory=args.data,
                     output_directory=args.output,
                     tta=args.tta,
                     batch_size=args.batch,
                     overwrite=args.overwrite,
                     iter=args.iter,
                     gpus=args.gpu)
    elif args.command == 'select_tilts':
            import easymode.tiltfilter.inference as tiltfilter
            tiltfilter.dispatch(input_tiltstack=args.tiltstack,
                                input_tomostar=args.tomostar,
                                input_xml=args.xml,
                                output_directory=args.output,
                                tta=args.tta,
                                gpus=args.gpu,
                                overwrite=args.overwrite,
                                extension=args.extension,
                                threshold=args.threshold)
    elif args.command == 'segment':
        features = [f.lower() for f in args.features]
        if args.use_2d:
            from easymode.core.ais_wrapper import dispatch_segment as dispatch_segment_2d
            for feature in features:
                dispatch_segment_2d(
                    feature=feature,
                    data_directory=args.data,  # now a list of patterns/paths
                    output_directory=args.output,
                    tta=args.tta,
                    batch_size=args.batch,
                    overwrite=args.overwrite,
                    data_format=args.format,
                    gpus=args.gpu
                )
        else:
            from easymode.segmentation.inference import dispatch_segment as dispatch_segment_3d
            for feature in features:
                dispatch_segment_3d(
                    feature=feature,
                    data_directory=args.data,  # now a list of patterns/paths
                    output_directory=args.output,
                    tta=args.tta,
                    batch_size=args.batch,
                    overwrite=args.overwrite,
                    data_format=args.format,
                    gpus=args.gpu
                )
    elif args.command == 'report':
        from easymode.core.reporting import report
        report(volume_path=args.tomogram,
               model=args.model,
               contact=args.contact,
               comment=args.comment)
    elif args.command == 'pick':
        from easymode.core.ais_wrapper import pick
        pick(target=args.feature,
             data_directory=args.data,
             output_directory=args.output,
             threshold=args.threshold,
             spacing=args.spacing,
             size=args.size,
             binning=args.binning,
             tomostar=args.tomostar,
             filament=args.filament,
             per_filament_star_file=args.separate_filaments,
             filament_length=args.length,
             centroid=args.centroid,
             min_particles=args.min_particles)
    elif args.command == 'reconstruct':
        from easymode.core.warp_wrapper import reconstruct
        reconstruct(frames=args.frames,
                    mdocs=args.mdocs,
                    apix=args.apix,
                    dose=args.dose,
                    extension=args.extension,
                    tomo_apix=args.tomo_apix,
                    thickness=args.thickness,
                    shape=args.shape,
                    steps=args.steps,
                    halfmaps=args.halfmaps,
                    force_align=args.force_align)
    elif args.command == 'set':
        if args.cache_directory:
            if os.path.exists(args.cache_directory):
                cfg.edit_setting("MODEL_DIRECTORY", args.cache_directory)
                print(f'Set easymode model directory to {args.cache_directory}. From now on, networks weights will be downloaded to and searched for in this directory. You may have to move previously downloaded models to this new directory, or download them again.')
            else:
                print(f'Directory {args.cache_directory} could not be found. Reverting to the previous directory: {cfg.settings["MODEL_DIRECTORY"]}.')
        if args.aretomo3_path:
            if os.path.exists(args.aretomo3_path):
                cfg.edit_setting("ARETOMO3_PATH", args.aretomo3_path)
                print(f'Set AreTomo3 path to {args.aretomo3_path}.')
            else:
                print(f'Path {args.aretomo3_path} could not be found. Reverting to the previous path: {cfg.settings["ARETOMO3_PATH"]}.')
        if args.aretomo3_env:
            cfg.edit_setting("ARETOMO3_ENV", args.aretomo3_env)
            print(f'Set AreTomo3 environment command to {args.aretomo3_env}.')
    elif args.command == 'package':
        from easymode.core.packaging import package_checkpoint
        package_checkpoint(title=args.checkpoint_directory.strip() if args.title is None else args.title, checkpoint_directory=args.checkpoint_directory)
    elif args.command == 'list':
        from easymode.core.distribution import list_remote_models
        list_remote_models()

if __name__ == "__main__":
    main()


