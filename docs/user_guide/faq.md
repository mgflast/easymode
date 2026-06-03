# FAQ

??? question "1. Inference is slow — much slower than 1–2 minutes per volume."

    The most likely cause is that easymode didn't find a GPU to use. Try running `easymode segment` without an explicit value for the `--gpu` argument. easymode will then auto-detect the available GPUs and use all of them. If none are found, a yellow warning message is printed.

    When easymode fails to find a GPU, the problem is usually a missing CUDA library. To see which tensorflow version you are on, run:

    ```
    pip list | grep tensorflow
    ```

    See [this chart](https://www.tensorflow.org/install/source#tested_build_configurations) to find out which CUDA library you need for that version of tensorflow. If you have a module system available on your cluster, you can usually load the required CUDA version that way.

    !!! note
        When using multiple GPUs, the workload is **distributed** across GPUs, not shared. The first segmentation will still only pop out after a few minutes — but a whole dataset will be processed much faster. If you only segment a single tomogram, multi-GPU has no effect: GPU 0 does the work and the rest sit idle.

??? question "2. Can I make processing any faster?"

    Try setting the test-time augmentation factor to 1: `--tta 1`. This will probably make the output worse, but is significantly faster. Once segmentation completes, you can inspect the results with Pom, create a subset of tomograms of interest, and then run an improved segmentation with:

    ```
    easymode segment x --data pom/subsets/subset_name.txt --tta 4
    ```

    to focus on just the relevant tomograms.

??? question "3. When I install Pom, easymode breaks."

    Sorry about that. There is an annoying compatibility problem between streamlit (used in Pom) and tensorflow (used in easymode). You can either:

    1. Use separate easymode and Pom environments, or
    2. After installing Pom, re-install easymode and then run `pip install protobuf==3.20.3`. There will be a warning telling you that this breaks tensorflow, but that is a lie. Ignore it.

??? question "4. Outputs for a particular network are very bad."

    There are a few different reasons why the networks can fail. If the output is 100% horrible, it may be an error in the code or the data format. If the output is bad, but not disastrously bad, the problem is more likely in the model itself.

    <b>1)</b> First of all, please inspect some of your tomograms manually. Are the features you want to segment at least somewhat visible to you, by eye? easymode was trained on non-inverted (dark features, light background — membranes are dark), Warp-reconstructed tomograms spanning a range of noise levels (half splits, raw tomograms, denoised tomograms). If your volumes are inverted or reconstructed very differently, easymode might fail.

    <b>2)</b> Input tomograms should be of a data type that the python module `mrcfile` is compatible with: `uint8`, `float32`, and `float16` work. We've seen things go wrong with less common 8-bit formats. If this is the case for you, please [let us know](https://github.com/mgflast/easymode/issues).

    <b>3)</b> As much as we'd like for easymode to work every time, segmentation quality is super dependent on tomogram quality — sometimes a bit too much so. Actin segmentation, for example, is currently good when tomograms are very crisp, but can get pretty bad even for otherwise-decent-but-a-bit-on-the-high-thickness-side tomograms. As a rule of thumb: if you think you would be able to straightforwardly annotate the feature, segmentation should work, and if it doesn't that's our failure. If you yourself struggle to detect the feature, we still wish we could help, but sometimes the best way forward is to acquire clearer data. As always: no processing tricks ever beat going back to the bench and improving sample quality.

    <b>4)</b> If your data is (somewhat) clear and the output is very poor, that's the most interesting case and we would love to help out — please consider [submitting a tomogram](../training_collection.md), [posting on GitHub](https://github.com/mgflast/easymode/issues), or reaching out directly (mlast@mrclmb.ac.uk).

    !!! tip
        You can always try the alternative model — run `easymode list` to see which models are available. In many cases there is both a 3D and a 2D model. For actin, for example, the 2D model can be useful, but the downside is that the filament shape is worse: elongated in Z, making adjacent filaments overlap often.

??? question "5. I want to segment X, but there is no model for X."

    We plan to expand the model library over time, but it does take quite a bit of work to train and validate a model. We try to focus on features that are of common interest to many users — for example, we plan to add networks to segment vesicles, Golgi, and virus-like particles, features that we expect are relevant to many different projects. If the feature you would like to segment is common to many species or cell types and you have some example data, we might be able to generate a (preliminary) network for it. Please reach out!

    In many cryoET projects, however, the target of interest is something very particular and a bit niche, and realistically we will never have a network for that. If you do specifically want to segment such a feature, we suggest trying out [Ais](https://github.com/bionanopatterning/Ais).

??? question "6. I can't download weights, because my GPU node is not connected to the internet."

    By default, easymode downloads and searches for local weight files (`.h5`, `.scnm`) files in `~/easymode/`. You can change the cache directory using:

    ```
    easymode set --cache-directory path/to/cache/
    ```

    You can download weights manually from [https://huggingface.co/mgflast/easymode/tree/main](https://huggingface.co/mgflast/easymode/tree/main), or download them automatically by running `easymode segment x` elsewhere. If you download them to a directory to which the offline node also has access and set that directory as the cache directory for the offline node, that should solve the problem.

??? question "7. Should I denoise my tomograms before running easymode?"

    Yes — denoising definitely helps to improve segmentation output. easymode was trained on half splits, raw volumes, and denoised (noise2noise) tomograms. Some users have found that missing-wedge-corrected tomograms (with e.g. IsoNet2) also yield better outputs, but note that such volumes were not included during training.

??? question "8. What pixel size should my tomograms be at?"

    Technically it doesn't matter — easymode rescales automatically. The training collection spans data collected with a pixel size of 0.7 - 3.5 Å/px and we expect acceptable performance within this range. For tomograms that were acquired at very low magnification, e.g. > 5 Å/px, we imagine that the output quality could be worse, especially for fine grained features like actin filaments or ribosomes. 

??? question "9. Can I retrain or fine-tune a model on my own data?"

    In principle the required tools are available in easymode (`easymode train`, `easymode extract`) so you could do it, but this is explicitly not what easymode was designed for. If you are interested in segmenting some specific feature for which there is no pretrained network available, we would recommend using [Ais](https://github.com/bionanopatterning/Ais) to train a bespoke network instead. You can train 2D, 2.5D, or shallow 3D networks in Ais, and training something up just for your own data is much easier than training a general network with a much larger architecture.

??? question "10. easymode picked thousands of false positives — how do I tune the picking threshold?"

    Always inspect segmentation outputs before picking. In ChimeraX you can play with the threshold, and using the **Hide dust** tool you can also find a suitable value for the minimum particle size. Try this for a couple of different tomograms, then pick reasonable values and try picking again. Also make sure that the --spacing argument is set to an appropriate value. For example, for a roughly globular particle with a diameter of 250 Å, you should set the minimum spacing to at least ~125-150 Å to avoid placing multiple coordinates within one blob.

??? question "11. Can I run multiple feature models on the same tomogram in one pass?"

    Yes — just do:

    ```
    easymode segment ribosome membrane mitochondrion ...
    ```

    with whatever features you want.

??? question "12. How much disk space will the segmentations use?"

    The default output format is `uint8`, so segmented volumes will be 2x or 4x smaller than WarpTools reconstructed tomograms (depending on whether the tomograms dtype is 16- or 32-bit float). Currently, if you segment many features, that does consume a lot of disk space. After picking or after setting up the Pom app, you can delete the segmented volumes to free up storage.
