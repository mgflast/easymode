# General denoisers

When you don't have the time, memory, or raw frames needed to train a dataset-specific denoiser, **easymode general denoisers** can denoise full tomograms directly and without any per-dataset training. These are pretrained networks that perform similar functionality to commonly used tools such as [cryoCARE](https://github.com/juglab/cryoCARE_pip), [DeepDeWedge](https://github.com/MLI-lab/DeepDeWedge), and others (work-in-progress on a general [IsoNet2](https://github.com/IsoNet-cryoET/IsoNet2) instance). All are invoked via `easymode denoise`, with the `--method` argument selecting which network to use.

```
easymode denoise --data warp_tiltseries/reconstruction --output warp_tiltseries/reconstruction/denoised --method n2n --gpu 0,1,2,3
```

Optional arguments:
```
--method {'n2n', 'ddw', 'iso'} Which pretrained network to use (default 'n2n'). See below.
--tta <int>                    Test-time augmentation factor (default: 1). When set to >1, the model denoises multiple augmented versions of the input and averages the results.
--iter <int>                   Number of denoising iterations to perform (default: 1). The denoiser can be re-applied to its own output to enhance contrast further -- at the risk of introducing artifacts.
--batch <int>                  Model batch size (default 1). Denoising uses 160x160x160 tiles and preserves the original hard 96x96x96 assembly.
--chunk-size <int>             Maximum number of tiles held for one streaming prediction call (default 16). This changes memory use, not tile geometry, and is reduced automatically after a TensorFlow memory error.
--overwrite                    If used, existing tomograms in --output are overwritten.
--gpu <string>                 Comma-separated list of GPU ids to use (default '0').
```

Denoising tiles are generated and reconstructed incrementally. This bounds the memory used by tile collections while preserving the original tile coordinates, zero padding, central-core crop, and hard placement of the legacy implementation.

!!! note "When to train your own denoiser"
    Both n2n and wedge-inpainting models learn to adapt to the specifics of your acquisition -- in n2n's case, to the particular noise statistics; for wedge inpainting, additionally to the missing-wedge geometry and the structural priors of your sample. A single general network can only approximate this, so for optimal denoising performance you should train a new network on your own data. And because denoising is unsupervised -- there is nothing to label, only compute to spend -- the cost of going custom is just GPU time.

    That said, for use within the easymode toolchain -- data inspection, segmentation, picking -- the pretrained general denoisers are perfectly adequate. They were in fact used during training of the segmentation networks, so applying them at inference time tends to *improve* segmentation results rather than hurt them.

## Method: `n2n` — noise2noise

The default method is our noise2noise (n2n, same framework as used in [cryoCARE](https://github.com/juglab/cryoCARE_pip)) model, trained on even/odd tomogram half-splits from 43 distinct sources. It is a pure denoiser: it removes shot noise but does not modify the missing wedge. We trained it in two stages so that it can be applied to a single full tomogram directly, no halfmaps required:

1. First, a standard even -> odd noise2noise UNet was trained.
2. That network was then used to denoise all training subtomograms, and a second UNet was trained to map a full subtomogram (even + odd) to the denoised output of the first network.

The resulting model approximates the result of true split-based denoising relatively well, is about twice as fast as running on splits, and removes the requirement that splittable data (e.g. raw frame stacks) be available. Because the network is essentially distilled from a strict noise2noise objective, it tends to preserve the contrast of genuine features faithfully.

## Method: `ddw` — DeepDeWedge

The `ddw` method uses a distilled [DeepDeWedge](https://github.com/MLI-lab/DeepDeWedge) network. Unlike n2n, DeepDeWedge jointly denoises **and** fills the missing wedge, which removes the elongation/streaking artefacts along the missing-wedge axis and tends to yield qualitatively cleaner volumes with more isotropic features.

The pretrained network was produced as follows:

1. A separate per-dataset DDW teacher network was trained on each of 48 datasets, using even/odd half-tomogram splits (mostly frame-based) and a missing wedge range appropriate for the dataset.
2. Each teacher was applied to its own dataset to produce "DDW-corrected" volumes.
3. A single general student network was then trained to map raw tomograms to the corresponding teacher-corrected volumes -- distilling all 48 per-dataset teachers into one general model.

The resulting network can be applied to a single full tomogram directly; halfmaps are not required.

!!! warning "Trade-off versus n2n"
    DeepDeWedge actively fills information that is not present in the raw data (the missing wedge). The training data was correspondingly augmented; per-dataset teachers were trained with simulated additional missing wedges so that the network has to learn to inpaint, rather than just to pass real data through. This makes the network more prone to **hallucinating features** and to **altering or reducing the contrast of genuine features** than a strict noise2noise model.

## Method: `iso` — IsoNet2

Distilled from [IsoNet2](https://github.com/IsoNet-cryoET/IsoNet2) — work in progress.
