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
--tile ZxYxX            Inference tile size for 3-D models (default 160x160x160).
--overlap <int>         Context discarded from each side of a 3-D tile (default 48; reduced automatically when required by the tile size).
--2d / --3d             Force the 2-D or 3-D model path.
```

For 3-D models, Easymode extracts, predicts, and reconstructs one tile at a time. This bounds host-memory use while preserving the existing tile coordinates, zero padding, retained central core, and hard placement of the legacy implementation. The Ais-backed 2-D path is unchanged.
!!! note
    easymode uses almost exactly the same network architecture as [Membrain](https://github.com/CellArchLab/MemBrain-v2). If your hardware can handle Membrain, it should also work for easymode.

!!! tip
    If accuracy is not _super_ important, set `--tta 1` to speed the segmentation up significantly. This can be useful when you're just exploring the data and only want to roughly estimate tomogram contents.

!!! note
    You can run multiple processes simultaneously on the same data and for the same feature, for example if you have some GPUs on node A and some more on node B. Whenever a thread starts processing a tomogram, it creates a provisional output file that prevents other threads from processing that same tomogram. In some cases, when you abort a process, a number of preliminary output files may not be deleted properly. In that case, you'll have to delete those files manually before restarting the segmentation.
    
    