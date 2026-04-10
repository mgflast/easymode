# Segmentation

Segmentation is the core of **easymode**. To list features for which pretrained models are available, run:

```
easymode list
```

Example output:
```
easymode can currently segment the following features:
    > ribosome
    > microtubule
    > mitochondrion
    > npc
    > tric
    > actin
    > mitochondrial_granule
    > cytoplasmic_granule
    > nuclear_envelope
    > vault
```

To segment any of these features in your tomograms, use the `easymode segment <feature>` command. For example:

```
easymode segment ribosome --data warp_tiltseries/reconstruction --output segmented/ --gpu 0,1,2,3,4,5,6,7
```

Optional arguments:
```
--tta <int>             Test-time augmentation factor (default: 4). The model will segment multiple augmented versions of the input and average the results.
--overwrite             If used, if output tomograms already exist in --output, they will be overwritten.
--format                Output format for the segmented volumes. Choices are 'float32', 'uint16', or 'int8' (default).
```
!!! note
    easymode uses almost exactly the same network architecture as [Membrain](https://github.com/CellArchLab/MemBrain-v2). If your hardware can handle Membrain, it should also work for easymode.

!!! tip
    If accuracy is not _super_ important, set `--tta 1` to speed up segmentation significantly. This can be useful when you're just exploring the data and only want to roughly estimate tomogram contents.

!!! note
    You can run multiple processes simultaneously on the same data and for the same feature, for example if you have some GPUs on cluster node A and some more on cluster node B. Whenever a thread starts processing a tomogram, it creates a provisional output file that prevents other threads from processing that same tomogram. In some cases, when you abort a process, a number of preliminary output files may not be deleted properly. In that case, simply delete those files manually before restarting the segmentation.
    
    