---
title: " "
---

<p align="center">
  <img src="../../assets/banner.png" alt="easymode logo" width="400">
</p>

Instructions for easymode [reconstruction](functions/preprocessing.md), [tilt selection](functions/preprocessing.md), [denoising](functions/preprocessing.md), and [segmentation](functions/segmentation.md).

## Scope and philosophy

In principle, the scope of easymode is pretty narrow: tomogram `.mrc` in, segmentation `.mrc` out. But by making biological features easily accessible, the idea is that easymode should enable and facilitate a lot of downstream processing tasks in cryoET. This includes picking segmented features to study them in their own right, filtering particles based on co-localization criteria, determining orientation priors (Euler angles) based on measurements of cellular context, or more generally defining searches for **particle identity, pose, and location based on biological and geometric priors**. Plus, in combination with [Pom](https://github.com/bionanopatterning/Pom), you can visualize and browse your dataset much more easily.

While we hope easymode helps solve the difficult first step of segmenting things of interest, it is difficult to build a general tool for how to then *use* these segmentations to solve your specific cryoET challenge. Once you get close to the specifics of a project, the requirements and questions also often become very specific.

## Examples and tutorials

To help with this, we include examples of using easymode for downstream tasks. We have tutorials for the demonstrations included in the paper:

1. **Subtomogram averaging** of [**ribosomes**](examples/ribosome.md), [**microtubules**](examples/microtubule.md), [**vault complexes**](examples/vault.md), and [**actin filaments**](examples/actin.md).
2. **Orientation correction** — picking [**nuclear pore complexes**](examples/npc_orientation.md) and correcting their orientation based on measurements of the cytoplasm and nucleus segmentations around them.
3. **Co-localization filtering** — filtering a heavily overpicked [**Hsp60–Hsp10**](examples/colocalization_filtering.md) complex particle set, derived from template matching, using co-localization and exclusion criteria for mitochondria, membranes, ribosomes, and ice particles.
4. **Lamella surface distance** — measuring the [**distances between ribosomes and the lamella surface**](examples/lamella_surface_distance.md), in order to select an undamaged particle subset for higher resolution averaging.

We also show a general example of [**combining easymode and Pom**](examples/easymode_and_pom.md). Using a small dataset of 100 of our own tomograms — for which we segment ribosomes, microtubules, membranes, mitochondria, and other features using easymode — we show how you can use Pom to render this segmented dataset searchable, how to visualize it, and how to select tomogram subsets for further analysis, all using almost entirely automated methods.

If you need help, feel free to [reach out](https://github.com/mgflast/easymode/issues). Or drop the link to these docs into your LLM of choice ask them; they are generally able to help (but watch out as they almost always mess up  cryoET coordinate/orientation conventions)