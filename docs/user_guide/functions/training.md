# Training

Although easymode is mainly about pretrained and general networks, you can also train your own network if you want. This requires you to have training labels - these can come from simulation, databases such as the CryoET Data Portal or EMPIAR, annotation in [Ais](https://mgflast.github.io/Ais/guide/annotating/), or any other source. If you already have data ready to go, training in easymode can be a good option.

If your goal is to train a network only for use on your own data, we strongly recommend training a network in Ais instead. We have [tutorials](https://mgflast.github.io/Ais/tutorials/quick_start/) available on training a network in Ais and using this for visualization and segmentation-based picking for subtomogram averaging. Ais training data files (`.scnt`) can also be used directly in easymode train.

## Data format
To train a network in easymode, training data is expected in a particular format. Inside some main directory - we'll call it `training_data` - easymode expects input subtomogram in sub-directories named `x_*/` (e.g. `x_raw/` and `x_isonet/`), labels in `y/`, and optionally a validity mask in `validity/`. Different files belonging to the same training sample should have the same filename. For example: `x_raw/sample001.mrc`, `x_isonet/sample001.mrc` and `y/sample001.mrc` would be a valid training sample with two input variants. Not all samples need to have all input variants.

Training input samples must be 3D .mrc files. Training labels may be 2D or 3D .mrc files. When they are 2D, they are assumed to be labels for the input samples' central slices (Z//2) only. All training samples within one training dataset are expected to have the same shape and assumed to have the same pixel size. Label voxels may have three values: 0 (background), 1 (foreground), and 2 (ignore label). When `validity/` masks are present, these must have values 0 (invalid) or 1 (valid). Wherever a validity mask has value 0, this voxel is ignored in the loss calculation. This is equivalent to directly setting the label value to 2. 

A minimum viable training dataset would be:

```
training_data/
├── x_main/<id>.mrc      
└── y/<id>.mrc           
```

A typical training dataset in easymode would be:

```
training_data/
├── x_raw/<id>.mrc      1000 160x160x160 .mrc files (full unfiltered tomograms)
├── x_even/<id>.mrc     1000 160x160x160 .mrc files (even half-splits)
├── x_odd/<id>.mrc      1000 160x160x160 .mrc files (odd half-splits)
├── x_n2n/<id>.mrc      1000 160x160x160 .mrc files (noise2noise-denoised)
├── x_iso/<id>.mrc      1000 160x160x160 .mrc files (iso-denoised)
├── x_ddw/<id>.mrc      1000 160x160x160 .mrc files (ddw-denoised)
├── y/<id>.mrc          1000 160x160x160 .mrc files (labels)
├── validity/<id>.mrc   1000 160x160x160 .mrc files (extraction validity masks)
└── metadata.json       
```

??? note "metadata.json"

    A dataset can describe itself in a `metadata.json` at its root. Every field is optional; easymode falls back to the .mrc headers and to sensible defaults when they are missing.

    ```json
    {
      "apix": 10.0,
      "apix_z": 10.0,
      "annotated_flavour": "x_n2n",
      "normalization": "global_mad"
    }
    ```

    `apix` and `apix_z` are the pixel sizes the model is packaged with, and determine how tomograms are rescaled at inference. `annotated_flavour` names the input variant that the labels were drawn on; it is the only one used for validation, and defaults to `x_main` if present or else to the first variant alphabetically. `normalization` is passed on to the trained model, so that inference measures the input the same way training did; `--normalization` on the command line overrides it.

??? note "Input normalization"

    Whatever normalization your training boxes were made with, inference has to repeat it. easymode cannot measure this from the boxes, so you declare it - with `--normalization`, or as `"normalization"` in `metadata.json` - and the model is tagged with your answer. Two schemes:

    ```
    global    the whole tomogram was scaled once, before the boxes were cut out of it. The default, and what all easymode models use.
    local     every box was scaled by its own statistic.
    ```

    Either way the statistic is center = mean, scale = MAD * 1.4826, measured over the central XY region of 32 evenly spaced Z slices. At inference a `global` model normalizes the tomogram once, before rescaling; a `local` model normalizes each tile instead. Prefer `global`: under `local`, a tile that happens to contain a large dense feature is scaled differently from one that does not, so the same density reaches the network differently depending on its surroundings. Getting this wrong is not visible in the training curves - it shows up as a model that segments much worse than its validation scores suggest.

## Training
With the data in place, training is:

```
easymode train --data training_data/ --title my_feature
```

Output:
```
training_data: 1000 samples, 160x160x160, 10.00 A/px
  flavours: x_ddw=1000 x_even=1000 x_iso=1000 x_n2n=1000 x_odd=1000 x_raw=1000 (annotated: x_n2n)
Loaded 950 samples for training (830 positive, 120 negative)
Loaded 50 samples for validation (44 positive, 6 negative)

Training model my_feature (arch: unet-membrain-groupnorm, crop: 160x160x160, 10.00 A/px, norm: global_mad)

Epoch 1/200
118/118 [==============================] - 611s 5s/step - loss: 0.8123 - precision: 0.3011 - recall: 0.5522 - val_loss: 0.7788
...
```

Arguments:
```
--data <str> [...]      One or more training datasets: a directory, a .tar archive, an Ais .scnt, or a glob pattern. Datasets are pooled; several can be listed after one --data.
-t, --title <str>       Name of the model, and of the resulting .h5 and .json.
-o, --output <str>      Directory for the model (default: ./{title}).
--normalization <str>   How the training boxes were normalized: global or local. Default: from metadata.json, else global. See above.
-e, --epochs <int>      Number of epochs (default: 200).
-b, --batch_size <int>  Batch size (default: 8).
-ls, --lr_start <float> Learning rate at the first epoch (default: 5e-3).
-le, --lr_end <float>   Learning rate at the last epoch (default: 5e-4). Some architectures impose their own schedule and ignore these.
--arch <str>            Network architecture (default: unet-membrain-groupnorm).
--size <ZxYxX>          Training crop shape. Each dimension must be divisible by the architecture's stride product: 8 for unet-easymode, 32 for unet-membrain*.
--apix <float>          Pixel size of the training data in Angstrom. Default: from metadata.json, else from the .mrc headers.
--weights <str>         Path to a .h5 to initialize training from, e.g. to fine-tune an easymode model on your own data.
--cache                 Keep every volume in RAM after its first read instead of re-reading it every epoch. Without this, training is IO bound and much slower. Prints the projected RAM requirement during initialization.
--xla                   Compile the network into optimized GPU code at the start of training. Typically several times faster; the first epoch is slow while the compilation runs.
--preview <int>         For debugging: write N augmented samples, as they would be served to the network, to the output directory instead of training (default 30, see below).
```

An archive passed to `--data` is unpacked next to itself for the duration of the run and cleaned up afterwards, so make sure there is room for a second copy of it. A directory is read in place.

## Loss function 
All available network architectures currently use a combined binary cross-entropy and DICE loss. Their weights are set with `--bce` and `--dice`.

## Augmentations
The following augmentations are used during training and their probabilities can be individually adjusted:

```
--aug_rot_xz_yz <float>  Continuous rotation around the X or Y axis, up to 15 degrees (default: 0.2).
--aug_rot_xy <float>     Continuous rotation around the Z axis, up to 22.5 degrees (default: 0.2).
--aug_flip <float>       Mirror flip along a random axis, Z, Y or X (default: 0.5).
--aug_blur <float>       Gaussian blur, sigma 0.5 to 1.0 (default: 0.2).
--aug_scale <float>      Magnification jitter, 90% to 110% (default: 0.2).
--aug_mixup <float>      Blend in up to 20% of a random all-background sample (default: 0.2).
--aug_contrast <float>   Contrast jitter, 80% to 125% (default: 0.5).
--aug_flavourmix <float> Fraction of samples served as a blend of two flavours rather than one pure flavour (default: 0.67).
```

Rotations by multiples of 90 degrees are always applied. A flip mirrors the sample, so if the feature you are training on is chiral, use `--aug_flip 0.0`.

To see what the network is actually being fed, use `--preview`. Instead of training, this writes N augmented samples and their labels to the output directory:

```
easymode train --data training_data/ --title my_feature --preview 100
```

```
my_feature_samples/
├── x_main/<hash>.mrc
└── y/<hash>.mrc
```


## Flavours

A flavour is one version of the same box: the raw subtomogram, an even/odd half-split, a Noise2Noise-denoised copy, an easymode general denoiser iso- or ddw-style variant, and so on. During training easymode serves either one pure flavour or a random blend of two, so that the network does not learn to depend on one particular preprocessing pipeline. This is what makes the pretrained easymode models work on raw and denoised tomograms alike. You do not need to supply multiple flavours.


## Using your model

Training writes `{title}.h5` and `{title}.json` into the output directory, rewriting the .h5 after every epoch that improves the loss - so a run that is stopped or killed still leaves a usable model. Point `easymode segment` at the .h5 — the .json next to it carries the pixel size, the architecture and the normalization scheme, so keep the two together. If you copy the `.h5` and `.json` into your easymode model cache directory (default: `~/easymode/`), the model also appears when you call `easymode list` and can be called with `easymode segment <your_model_name>`.

```
easymode segment --model my_feature/my_feature.h5 --data warp_tiltseries/reconstruction --output segmented/
```
