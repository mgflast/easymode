import argparse
import easymode.core.config as cfg
import os

# TODO: clear cache command

def main():
    parser = argparse.ArgumentParser(description="easymode: pretrained general networks for cryoET.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    train_parser = subparsers.add_parser('train', help='Train an easymode network.')
    train_parser.add_argument('-t', "--title", type=str, required=True, help="Title of the model.")
    train_parser.add_argument('-f', "--features", nargs="+", required=True, help="List of features to train on, e.g. 'Ribosome3D Junk3D' - corresponding data directories are expected in /cephfs/mlast/compu_projects/easymode/training/3d/data/{features}")
    train_parser.add_argument('-e', "--epochs", type=int, help="Number of epochs to train for (default 500).", default=500)
    train_parser.add_argument('-b', "--batch_size", type=int, help="Batch size for training (default 8).", default=8)
    train_parser.add_argument('-lr', "--lr_start", type=float, help="Initial learning rate for the optimizer (default 1e-3).", default=1e-4)
    train_parser.add_argument('-le', "--lr_end", type=float, help="Final learning rate for the optimizer (default 1e-5).", default=1e-5)

    set_params = subparsers.add_parser('set', help='Set environment variables.')
    set_params.add_argument('--cache_directory', type=str, help="Path to the directory to store and search for easymode network weights in.")
    set_params.add_argument('--aretomo3_path', type=str, help="Path to the AreTomo3 executable.")
    set_params.add_argument('--aretomo3_env', type=str, help="Command to initialize the AreTomo3 environment, e.g. 'module load aretomo/3.1.0'")


    package = subparsers.add_parser('package', help='Package model and weights. Note that this is used for 3D models only; 2D models are packaged and distributed with Ais.')
    package.add_argument('-t', "--title", type=str, required=True, help="Title of the model to package.")
    package.add_argument('-c', "--checkpoint_directory", type=str, required=True, help="Path to the checkpoint directory to package from.")
    package.add_argument('-ou', "--output_directory", type=str, default='/cephfs/mlast/compu_projects/easymode/training/3d/packaged/', help="Output directory to save the packaged model weights.")

    subparsers.add_parser('list', help='List the features for which pretrained general segmentation networks are available.')

    segment = subparsers.add_parser('segment', help='Segment data using pretrained easymode networks.')
    segment.add_argument("feature", metavar='FEATURE', type=str, help="Feature to segment. Use 'easymode list' to see available features.")
    segment.add_argument("--data", type=str, required=True, help="Directory containing .mrc files to segment.")
    segment.add_argument('--gpu', required=True, type=str, help="Comma-separated list of GPU IDs to use (e.g., 0,1,3,4)")  #
    segment.add_argument('--tta', required=False, type=int, default=4, help="Integer between 1 and 16 (or max 8 for 2d). For values > 1, test-time augmentation is performed by averaging the predictions of several transformed versions of the input. Higher values can yield better results but increase computation time. (default: 4)")
    segment.add_argument('--output', required=True, type=str, help="Directory to save the output")
    segment.add_argument('--overwrite', action='store_true', help='If set, overwrite existing segmentations in the output directory.')
    segment.add_argument('--batch', type=int, default=1, help='Batch size for segmentation (default 1). Volumes are processed in batches of 160x160x160 shaped tiles. In/decrease batch size depending on available GPU memory.')
    segment.add_argument('--format', type=str, choices=['float32', 'uint16', 'int8'], default='int8', help='Output format for the segmented volumes (default: float32). Choose uint16 or int8 to save disk space, but note that this may reduces the precision of the output (which should hardly matter).')

    pick = subparsers.add_parser('pick', help='Pick particles in segmented volumes.')
    pick.add_argument("feature", metavar='FEATURE', type=str, help="Feature to pick, based on segmentations.")
    pick.add_argument('--data', required=True, type=str, help="Path to directory containing input .mrc's.")
    pick.add_argument('--output', required=False, type=str, default=None, help="Directory to save output coordinate files to. If left empty, will save to the input data directory.")
    pick.add_argument('--threshold', required=False, type=float, default=128, help="Threshold to apply to volumes prior to finding local maxima (default 128). Regardless of the segmentation .mrc dtype, the value range is assumed to be 0-255.")
    pick.add_argument('--binning', required=False, type=int, default=2, help="Binning factor to apply before processing (faster, possibly less accurate). Default is 2.")
    pick.add_argument('--spacing', required=False, type=float, default=10.0, help="Minimum distance between particles in Angstrom.")
    pick.add_argument('--size', required=False, type=float, default=10.0, help="Minimum particle size in cubic Angstrom.")
    pick.add_argument('--no_tomostar', dest='tomostar', action='store_false', help='Include this flag in order NOT to rename tomograms in the .star files from etc_10.00Apx.mrc to etc.tomostar.')

    reconstruct = subparsers.add_parser('reconstruct', help='Reconstruct tomograms using WarpTools and AreTomo3.')
    reconstruct.add_argument('--frames', type=str, required=True, help="Directory containing raw frames.")
    reconstruct.add_argument('--mdocs', type=str, required=True, help="Directory containing mdocs.")
    reconstruct.add_argument('--apix', type=float, required=True, help="Pixel size of the frames in Angstrom.")
    reconstruct.add_argument('--dose', type=float, required=True, help="Dose per frame in e-/A^2.")
    reconstruct.add_argument('--extension', type=str, default=None, help="File extension of the frames (default: auto).")
    reconstruct.add_argument('--tomo_apix', type=float, default=10.0, help="Pixel size of the tomogram in Angstrom (default: 10.0). Easymode networks were all trained at 10.0 A/px.")
    reconstruct.add_argument('--thickness', type=float, default=3000.0, help="Thickness of the tomogram in Angstrom (default: 3000).")
    reconstruct.add_argument('--shape', type=str, default=None, help="Frame shape (e.g. 4096x4096). If not provided, the shape is inferred from the data.")
    reconstruct.add_argument('--steps', type=str, default='1111111', help="7-character string indicating which processing steps to perform (default: '1111111'). Each character corresponds to a specific step: 1 to perform the step, 0 to skip it. The steps are: 1) Frame motion and CTF, 2) Importing tilt series, 3) Creating tilt stacks, 4) Tilt series alignment, 5) Import alignments, 6) Tilt series CTF, 7) Reconstruct volumes.")
    reconstruct.add_argument('--no_halfmaps', dest='halfmaps', action='store_false', help="If set, do not generate half-maps during motion correction or tomogram reconstruction. This precludes most methods of denoising.")

    denoise = subparsers.add_parser('denoise', help='Denoise or enhance contrast of tomograms.')
    denoise.add_argument('--data', type=str, required=True, help="Directory containing tomograms to denoise. In mode 'splits', this directory is expected to contain two subdirectories 'even' and 'odd' with the respective tomogram splits.")
    denoise.add_argument('--output', type=str, required=True, help="Directory to save denoised tomograms to.")
    denoise.add_argument('--mode', type=str, choices=['splits', 'direct'], help="Denoising mode. splits: statistically sound denoising of independent even/odd splits. direct: denoise the complete tomogram using a network that was trained on even/odd split denoised data. This last option helps avoid having to generate even/odd frame and volume splits.")
    denoise.add_argument('--tta', type=int, default=1, help="Test-time augmentation factor (default 1). Input volumes can be processed multiple times in different orientations and the results averaged to yield a (potentially) better result. Higher values increase computation time. Maximum is 16, default is 1.")
    denoise.add_argument('--gpu', required=True, type=str, help="Comma-separated list of GPU IDs to use (e.g., 0,1,3,4)")  #
    denoise.add_argument('--overwrite', action='store_true', help='If set, overwrite existing segmentations in the output directory.')
    denoise.add_argument('--batch', type=int, default=1,help='Batch size for segmentation (default 1). Volumes are processed in batches of 128x128x128 shaped tiles. In/decrease batch size depending on available GPU memory.')
    denoise.add_argument('--iter', type=int, default=1, help="Only valid in direct mode: number of denoising iterations to perform (default 1). If you are really starved for contrast, try increasing this - but beware of artifacts.")


    denoise_train = subparsers.add_parser('denoise_train', help='Train a denoising network.')
    denoise_train.add_argument('--mode', type=str, choices=['splits', 'direct'], required=True, help="Denoising type. splits: statistically sound denoising of independent even/odd splits. direct: denoise the complete tomogram using a network that was trained on even/odd split denoised data. This last option helps avoid having to generate even/odd frame and volume splits.")
    denoise_train.add_argument('--extract', action='store_true', help="If set, perform step 1 of the training: extraction of subvolumes.")
    denoise_train.add_argument('--train', action='store_true', help="If set, perform step 2 of the training: the actual training run.")
    denoise_train.add_argument('--n', type=int, default=10, help="Number of samples to use per tomogram (default 10).")
    denoise_train.add_argument('--epochs', type=int, help="Number of epochs to train for (default 500).", default=200)
    denoise_train.add_argument('--batch', type=int, help="Batch size for training (default 16).", default=16)
    denoise_train.add_argument('--box',  type=int, help="Box size for training (default 96).", default=96)
    denoise_train.add_argument('--ls', type=float, help="Initial learning rate for the optimizer (default 1e-3).", default=1e-4)
    denoise_train.add_argument('--le', type=float, help="Final learning rate for the optimizer (default 1e-5).", default=1e-5)

    args, unknown = parser.parse_known_args()

    if args.command == 'train':
        from easymode.core.train import train_model
        train_model(title=args.title,
                    features=args.features,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr_start=args.lr_start,
                    lr_end=args.lr_end
                    )
    elif args.command == 'denoise':
        from easymode.n2n.inference import dispatch_segment
        dispatch_segment(mode=args.mode,
                         input_directory=args.data,
                         output_directory=args.output,
                         tta=args.tta,
                         gpus=args.gpu,
                         batch_size=args.batch,
                         overwrite=args.overwrite,
                         iter=args.iter)
    elif args.command == 'denoise_train':
        if args.extract:
            from easymode.n2n.train import N2NDatasetGenerator
            dataset = N2NDatasetGenerator(mode=args.mode, samples_per_tomogram=args.n)
            dataset.generate()
        if args.train:
            from easymode.n2n.train import train_denoiser
            train_denoiser(mode=args.mode,
                           batch_size=args.batch,
                           box_size=args.box,
                           epochs=args.epochs,
                           lr_start=args.ls,
                           lr_end=args.le)
        if not args.extract and not args.train:
            print("Neither --extract or --train flag set!.")
    elif args.command == 'segment':
        from easymode.core.inference import dispatch_segment
        dispatch_segment(feature=args.feature.lower(),
                data_directory=args.data,
                output_directory=args.output,
                tta=args.tta,
                gpus=args.gpu,
                batch_size=args.batch,
                overwrite=args.overwrite,
                data_format=args.format)
    elif args.command == 'pick':
        from easymode.core.pick import pick
        pick(target=args.feature.lower(),
             data_directory=args.data,
             output_directory=args.output if args.output is not None else args.data,
             spacing=args.spacing,
             size=args.size,
             binning=args.binning,
             tomostar=args.tomostar)
    elif args.command == 'reconstruct':
        from easymode.core.warp import reconstruct
        reconstruct(frames=args.frames,
                    mdocs=args.mdocs,
                    apix=args.apix,
                    dose=args.dose,
                    extension=args.extension,
                    tomo_apix=args.tomo_apix,
                    thickness=args.thickness,
                    shape=args.shape,
                    steps=args.steps,
                    halfmaps=args.halfmaps)
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
        package_checkpoint(title=args.title, checkpoint_directory=args.checkpoint_directory, output_directory=args.output_directory)
    elif args.command == 'list':
        from easymode.core.distribution import list_remote_models
        list_remote_models()

if __name__ == "__main__":
    main()


