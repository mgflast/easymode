---
title: " "
---

`easymode segment membrane`

The membrane model was trained on a combination of Membrain-seg output labels and Ais 2D membrane network output labels. Aside from the labelled training samples, we included around 500 training subtomograms that did not contain any membranes but did contain features that are commonly erroneously segmented as membranes: microtubules, carbon edges, actin bundles, intermediate filaments, that sort of stuff. The idea was to improve slightly upon Membrain-seg by training out these errors. 

That was somewhat succesful; in our experience / qualitative assessment, easymode membrane segmentation does a better job at not labelling microtubules and other dense membrane-like features. However, it does not do any better at labelling membranes themselves. Probably a bit worse, in fact, as the membrane-segmentation-specific loss function used in Membrain-seg plus their missing wedge augmentation are likely more suitable for membranes.

Especially for membrane segmentation there are a number of pretrained networks available. If you're looking for the best possible segmentation accuracy you might want to have a look at those - examples are [Membrain-seg](https://www.biorxiv.org/content/10.1101/2024.01.05.574336v2.abstract), [TARDIS](https://www.biorxiv.org/content/10.1101/2024.12.19.629196v1.full.pdf), and [Seghiri & Gallego Nicolas et al.](https://www.biorxiv.org/content/10.64898/2026.01.15.699326v1.full.pdf).

**Example output**
<br>
<p style="text-align:center;">
  <video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:16/9; background:#fff; border-radius:8px; display:block; margin:auto;">
    <source src="../../assets/membrane.mp4" type="video/mp4">
    Video failed to load.
  </video>
</p>
Example of `easymode segment membrane` output overlaid on a tomogram from EMPIAR-11943 (FIB-milled D. discoideum).
