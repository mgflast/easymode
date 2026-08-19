# General denoisers

When you don't have the time, memory, or raw frames needed to train a dataset-specific denoiser, **easymode general denoisers** can denoise full tomograms directly and without any per-dataset training. These are pretrained networks that perform similar functionality to commonly used tools such as [cryoCARE](https://github.com/juglab/cryoCARE_pip), [DeepDeWedge](https://github.com/MLI-lab/DeepDeWedge), and [IsoNet2](https://github.com/IsoNet-cryoET/IsoNet2). All are invoked via `easymode denoise`, with the `--method` argument selecting which network to use.

```
easymode denoise --data warp_tiltseries/reconstruction --output warp_tiltseries/reconstruction/denoised --method n2n --gpu 0,1,2,3
```

Optional arguments:
```
--method {'n2n', 'ddw', 'iso'} Which pretrained network to use (default 'n2n'). See below.
--tta <int>                    Test-time augmentation factor (default: 1, maximum 16). When set to >1, the model denoises multiple augmented versions of the input and averages the results.
--iter <int>                   Number of denoising iterations to perform (default: 1). The denoiser can be re-applied to its own output to enhance contrast further -- at the risk of introducing artifacts.
--batch <int>                  Batch size (default 1). Volumes are processed in batches of 128x128x128 tiles.
--overwrite                    If used, existing tomograms in --output are overwritten.
--gpu <string>                 Comma-separated list of GPU ids to use (default '0').
```

!!! warning "Use raw reconstructions"
    All models in the easymode collection were trained on Warp reconstructions, and for the denoisers this matters a lot: run them on '*raw*' reconstructions. Anything you do to a tomogram beforehand - low pass filtering, deconvolution, a previous round of denoising - takes it out of the training distribution, and the denoiser is unlikely to improve it. 

!!! note "When to train your own denoiser"
    A network trained on your own data will always adapt better to your noise statistics, missing wedge geometry, and structural priors than a general one can, and since denoising is unsupervised the only cost of going custom is GPU time.

    That said, for use within the easymode toolchain - data inspection, segmentation, picking - the pretrained denoisers are perfectly adequate. They were in fact used during training of the segmentation networks, so applying them at inference time tends to *improve* segmentation results rather than hurt them.

## The methods

`n2n` is the default and is a noise2noise model, the same framework as used in [cryoCARE](https://github.com/juglab/cryoCARE_pip). It is a pure denoiser: it removes shot noise but does not modify the missing wedge, and because it is trained against a strict noise2noise objective it tends to preserve the contrast of genuine features faithfully. `ddw` ([DeepDeWedge](https://github.com/MLI-lab/DeepDeWedge)) and `iso` ([IsoNet2](https://github.com/IsoNet-cryoET/IsoNet2)) instead denoise **and** fill the missing wedge. Method `iso` in particular tends to produce very clean tomograms. Both `n2n` and `iso` type tomograms were used for training of all segmentation models.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1em;">
<div>
<video autoplay loop muted playsinline controls style="width:100%; background:#fff; border-radius:8px; display:block;">
  <source src="../../../assets/denoise_raw.mp4" type="video/mp4">
  Video failed to load.
</video>
<p style="text-align:center;"><code>raw</code></p>
</div>
<div>
<video autoplay loop muted playsinline controls style="width:100%; background:#fff; border-radius:8px; display:block;">
  <source src="../../../assets/denoise_n2n.mp4" type="video/mp4">
  Video failed to load.
</video>
<p style="text-align:center;"><code>n2n</code></p>
</div>
<div>
<video autoplay loop muted playsinline controls style="width:100%; background:#fff; border-radius:8px; display:block;">
  <source src="../../../assets/denoise_ddw.mp4" type="video/mp4">
  Video failed to load.
</video>
<p style="text-align:center;"><code>ddw</code></p>
</div>
<div>
<video autoplay loop muted playsinline controls style="width:100%; background:#fff; border-radius:8px; display:block;">
  <source src="../../../assets/denoise_iso.mp4" type="video/mp4">
  Video failed to load.
</video>
<p style="text-align:center;"><code>iso</code></p>
</div>
</div>

## How they were trained

Used conventionally, all three methods have to be trained per dataset and need even/odd half-splits as input. We removed both requirements by distillation: a conventionally trained instance is used to generate raw/denoised pairs, and a single network is then trained on those pairs to reproduce the same result from a raw tomogram. The architecture of the three released networks is identical; only the weights differ. All three were also trained with similar  sampling, taking the same number of boxes from every dataset so that no single source dominates.

For `n2n` the training was general from the beginning: we first trained a single residual noise2noise-style UNet on even/odd pairs from 43 unique datasets. We then used that even/odd denoiser to generate raw/denoised pairs, and trained a second network on those. This approximates split-based denoising relatively well, is about twice as fast as running on splits, and removes the requirement that splittable data be available at all.

`ddw` and `iso` were made by teacher-student distillation instead. We trained separate DeepDeWedge and IsoNet2 teacher networks on each of 48 half-split datasets, then applied every resulting network to its own training pairs to generate raw/denoised pairs. A single student network was then trained on the pairs from all 48 teachers. The DeepDeWedge teachers were given a missing wedge range appropriate for each dataset; because IsoNet2 accepts a missing wedge value per tomogram, the IsoNet2 teachers were given the correct geometry for every individual tomogram.

!!! note "Blurry patches"
    Both wedge-inpainting methods occasionally output locally blurred patches. Our guess at the cause: in noise2noise the training pairs are static, but DDW and IsoNet2 periodically replace their training targets with their own current best attempt at inpainting. Since both learn denoising and inpainting at the same time, a *lazy denoising solution* can creep into the targets as they are updated -- a low pass filter is, after all, a decent way to remove noise -- and then reinforce itself. n2n cannot land there, because its pairs are never edited.

    Whether this is a problem in your project depends on a lot of things. As always, inspect results from automated processing steps to determine whether it has worked well enough for whatever your purpose with the data is.

!!! note "What these networks are for"
    We originally made these networks so that we could apply different denoiser-style augmentations on the fly during training of the segmentation networks. We did not optimize them for perfect denoising quality -- at this scale, we can't. We find that they give qualitatively good denoising and we now use the general denoisers rather than training bespoke instances.

    When using these denoisers, please credit the authors of the original methods: [cryoCARE](https://doi.org/10.1109/ISBI.2019.8759519), [DeepDeWedge](https://doi.org/10.1038/s41467-024-51438-y), and [IsoNet](https://doi.org/10.1038/s41467-022-33957-8).
