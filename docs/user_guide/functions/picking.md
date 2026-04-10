# Picking

easymode wraps around [Ais](https://github.com/bionanopatterning/Ais) to turn segmented volumes into coordinate .star files.

## Globular particles
For approximately globular particles (such as ribosomes) use:

```
easymode pick ribosome --data segmented --output coordinates/ribosome --binning 3 --size 2000000 --spacing 250
```
Output:
```
Found 500 files with pattern /cephfs/mlast/easymode/testing/yeast/segmented/*__ribosome.mrc. Picking in blob mode.

1/5 (process 19) - 0 particles in 20240802_l25t10_10.00Apx__ribosome.mrc
1/5 (process 7) - 3 particles in 20240802_l15t04_4_10.00Apx__ribosome.mrc
1/5 (process 14) - 194 particles in 20240802_l10t01_2_10.00Apx__ribosome.mrc
1/5 (process 12) - 39 particles in 20240802_l15t01_3_10.00Apx__ribosome.mrc
...

found 109081 particles in total.
```

Arguments:
```
--output <str>          Directory to output per-tomogram .star files (will be created if it doesn't exist)
--threshold <float>     Value to threshold segmentation volumes at (default: 128). Value range is interpreted as [0, 255], regardless of the segmentation volume data type.
--binning <int>         Factor by which the segmentation volumes were binned relative to the original tomograms (default: 3). Higher value is much faster, but for small particles may introduce inaccuracies.
--size <int>            Minimum volume of non-zero regions in the thresholded segmentation, in cubic Å.
--spacing <int>         Minimum spacing between picked coordinates, in Å.
--no_tomostar           Include this flag in order NOT to convert Warp-style tomogram names (tomogram_10.00Apx.mrc) into Warp-style tomogram starfile names (tomogram.tomostar). 
--centroid              Include this flag in order to pick the centroid of connected components, rather than maxima in the distance transform. Useful for irregularly shaped particles. Use only if individual particles are well separated in the segmentation!
--min_particles <int>   Save .star files only for tomograms containing at least this number of particles.   
```

!!! note
    easymode assumes that for a feature 'feature', the segmented tomograms are named `tomogram__feature.mrc`. easymode also saves volumes using this pattern when running `easymode segment`.

## Filaments
For picking along filaments, for instance along microtubules, we use the new filament tracing function in Ais. Simply include the `--filament` flag:
```
easymode pick microtubule --data segmented --output coordiantes/microtubule --binning 3 --length 2000 --spacing 82 --separate_filaments  
```

```
Found 1266 files with pattern /cephfs/mlast/easymode/testing/human/segmented/*__microtubule.mrc. Picking in filament mode.

1/11 (process 89) - 40 particles in 1 filaments in 250715_l1p3_10.00Apx__microtubule.mrc
1/11 (process 46) - 0 particles in 0 filaments in l14p5_5_10.00Apx__microtubule.mrc
1/12 (process 24) - 102 particles in 6 filaments in 250106_l6p1_6_10.00Apx__microtubule.mrc
...

found 28381 particles in total. that's 232.7 um of microtubule :)
```
!!! tip
    Use the flag `--separate_filaments` to save .star files per filament, rather than per tomogram. This is useful if you want to do per-filament averaging, which can be used to assign polarity or count protofilament numbers.

!!! note
    Segmentation is computationally expensive part, but turning segmentations into coordinates is quite cheap. The examples above, picking in 500 and 1266 tomograms respectively, both completed in less than a minute. This is a typical speed if the `--binning` value is set to 3 or higher and the filesystem can keep up with all the loading. 