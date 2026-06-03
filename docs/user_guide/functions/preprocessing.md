# Tomogram reconstruction

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
!!! tip "Tip: skip the halfmaps"
    If you don't plan to use the noise2noise denoiser, or you intend to use the DeepDeWedge denoiser (which works on a single tomogram), you can save time and memory by skipping the generation of halfmaps with `--no_halfmaps`. See the [general denoisers](general_denoisers.md) page for details.

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
