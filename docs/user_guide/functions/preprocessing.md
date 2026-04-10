# Preprocessing

Although the core of **easymode** is really just about feature detection, we've included a couple of tools to facilitate cryoET data preprocessing. These are built on [Warp](https://warpem.github.io) and [AreTomo3](https://github.com/czimaginginstitute/AreTomo3).

## Fully automated tomogram reconstruction

Use `easymode reconstruct` to convert input frames and mdoc files into tomograms fully automatically. This command simply wraps around Warp and AreTomo3, so all credit goes to those authors! Requires setting up Warp and AreTomo3 as described on the [installation](installation.md) page.

```
easymode reconstruct --frames frames/ --mdocs mdocs/ --apix 1.56 --dose 3.20
```

Optional arguments:
```
--thickness <int>         Thickness of the tomogram in Å (default: 3000)
--extension <str>         File extension of the input frames (default: auto-detected)
--tomo_apix <float>       Output pixel size in Å (default is 10 Å/px).
--shape <int>x<int>       Size (in pixels) of the tilt images (default: auto-detected)
--no_halfmaps             Include this flag in order NOT to generate even/odd frame and volume splits
```
!!! tip "Tip: direct denoising"
    If you don't need _perfect_ denoising performance and _good_ is good enough, you can use a pretrained denoiser and save time & memory by skipping the generation of halfmaps with `--no_halfmaps`. More about this in the **Denoising** section.

## Tilt selection
Use `easymode select_tilts` to automatically identify bad tilt images and exclude them from tomogram reconstruction. To avoid repeating tomogram reconstruction, you could run the above `easymode reconstruct` with argument `--steps 11100000` first, then run tilt selection, and then reconstruct with `--steps 00011111`. Or just reconstruct and think about tilt selection later.
```
easymode select_tilts --tomostar tomostar/ --output tomostar/selected_tilts/
```

Optional arguments:
```
--tiltstack <string>       Path to a single tilt stack file OR glob pattern for multiple tilt stacks (*.st) OR directory containing tilt stacks (default: warp_tiltseries/tiltstacks/).
--tomostar <string>        Directory containing Warp-style tomogram star files.     
--xml <string>             Path to a directory containing Warp-style tilt series .xml files. The 'UseTilt' parameter in these files will be updated based on the tilt selection (default: warp_tiltseries/xml/).
--tta <int>                Test-time augmentation factor (default: 1, max 8). The model will predict multiple augmented versions of each tilt image and average the results.
--gpu <int>                GPU id(s) to use for inference.
--extension <str>          Extension of the tomogram star files in --tomostar. Default is '*.tomostar'.
--threshold <float>        Threshold for tilt exclusion (default: 0.5). Tilt images with a predicted quality score below this value will be excluded. Score range is 1.0 to 0.0, with 1.0 being good.
--overwrite                If used, if output star files already exist in --output, they will be overwritten. 
```
If you provide --tomostar only, a standard Warp file structure is assumed, with tilt stacks in warp_tiltseries/tiltstack/\*/\*.st and xml files in warp_tiltseries/*.xml. If you provide --tiltstack only, output tomostar files will be written in the location of each tilt stack file and no Warp-style file structure is used. 

## Denoising
In cases where you don't have access to raw frames, or you want to save memory, or want to save time, you can denoise your full tomograms directly using `easymode denoise`, our noise2noise model pretrained on data from 43 distinct sources. 

```
easymode denoise --data warp_tiltseries/reconstruction --output warp_tiltseries/reconstruction/denoised --mode direct --method n2n --gpu 0,1,2,3
```

Optional arguments:
```
--mode {'direct', 'splits'}     Use 'direct' mode to denoise full tomograms, or 'splits' to denoise and combine independent even and odd tomogram splits.
--tta <int>                     Test-time augmentation factor (default: 1). Probably superflous for denoising; when set to >1, the model will denoise multiple augmented versions of the input and average the results.
--iter <int>                    Number of denoising iterations to perform (default: 1). Only valid in 'direct' mode. A direct denoiser can be re-applied to its own output, to enhance contrast further - at the risk of introducing many artifacts.
--overwrite                     If used, if output tomograms already exist in --output, they will be overwritten.
```

!!! note
    For optimal denoising , it is probably always best to train a new network on your own data. But for general segmentation and picking purposes, the easymode general denoiser does the job. 


### Direct versus split denoising

We currently offer one denoising method (noise2noise), and two denoising modes: direct and splits. In 'splits' mode, we denoise in the usual way: using a network that was trained on independent data splits (e.g., even and odd volumes), running inference on the even and odd volume and averaging the results. This is the statistically sound and superior method, but generating and processing the independent splits can cost a lot of time and memory. Plus, it requires access to splittable data, ideally raw frame stacks, which are not always available (although you can always split on the tilt angles)

In 'direct' mode, we use a network that was trained in two steps. First, we trained in the typical even -> odd fashion. We then applied the resulting network to denoise all training subtomograms, and prepared a new training set using full subtomograms (i.e. the even + odd splits) as the input, and the denoised subtomogram as the output. The resulting model can be applied to full tomograms directly, and approximates the output that even/odd denoising achieves relatively well. It about twice as fast and doesn't require data splitting, which can help save a lot of memory.
