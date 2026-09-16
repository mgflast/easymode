# Segmentation

Segmentation is the core of **easymode**. To list features for which pretrained models are available, run:

```
easymode list
```

Example output:
```
easymode can currently segment the following features:

                                    versions
   > actin                          sv2-3d*, sv2-2.5d
   > atp_synthase                   sv2-2d*
   > chromatin                      sv2-2d*
   ...
   > ribosome                       sv3-3d*, sv2-3d
   > void                           sv3-3d*, sv2-2d

   *default model. use --version to select a specific model variant.
```

Each feature can have more than one model. The version tag says how a model was trained and what it is: `sv2` or `sv3` is the supervision, i.e. whether the training labels were 2D or 3D; `2d`, `2.5d` or `3d` is the network. A 2.5D model has a 2D architecture but takes a slab of slices as input rather than a single slice. The default is marked with `*`; pick another with `easymode segment <feature> --version <tag>`.

To segment any of these features in your tomograms, use the `easymode segment <feature>` command. For example:

```
easymode segment ribosome --data warp_tiltseries/reconstruction --output segmented/ --gpu 0,1,2,3,4,5,6,7
```

Optional arguments:
```
--version <tag>         Which version of the model to use (see `easymode list`). Default: the feature's default model.
--tta <int>             Test-time augmentation factor (default: 4). The model will segment multiple augmented versions of the input and average the results.
--overwrite             If used, if output tomograms already exist in --output, they will be overwritten.
--format                Output format for the segmented volumes. Choices are 'float32', 'uint16', or 'int8' (default).
```
!!! note
    easymode uses almost exactly the same network architecture as [Membrain](https://github.com/CellArchLab/MemBrain-v2). If your hardware can handle Membrain, it should also work for easymode.

!!! tip
    If accuracy is not _super_ important, set `--tta 1` to speed the segmentation up significantly. This can be useful when you're just exploring the data and only want to roughly estimate tomogram contents.

!!! note
    You can run multiple processes simultaneously on the same data and for the same feature, for example if you have some GPUs on node A and some more on node B. Whenever a thread starts processing a tomogram, it creates a provisional output file that prevents other threads from processing that same tomogram. In some cases, when you abort a process, a number of preliminary output files may not be deleted properly. In that case, you'll have to delete those files manually before restarting the segmentation.
    
    