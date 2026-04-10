---
title: " "
---

`easymode segment void`

_Void_ refers to regions of tomograms that are outside the sample (e.g. above or below the lamella), contain poor-quality data (e.g. thick or poorly reconstructed areas), or are dominated by artefacts (e.g. dense ice contamination). One application of segmenting void is to measure or **approximate the distance from a particle to the edges of a lamella**; sampling the void segmentation output at the location of picked particles offers a practical way of estimating this distance. 

A second use of the void model is to **approximate the quality of entire tomograms**, as demonstrated in [this preprint](https://www.biorxiv.org/content/10.1101/2025.01.16.633326v1). Simply running `easymode segment void` on all tomograms in a dataset, tabulating the average void output value for each tomogram, and sorting low to high is a practical way to filter good from worse tomograms.

The model is relatively fast. It works at 30 Å/px and most tomograms are segmented in a few seconds.



**Example output**
<br>
<p style="text-align:center;">
  <video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:16/9; background:#fff; border-radius:8px; display:block; margin:auto;">
    <source src="../../assets/void.mp4" type="video/mp4">
    Video failed to load.
  </video>
</p>
Example of `easymode segment void` output overlaid on a tomogram from EMPIAR-11899 (FIB-milled D. discoideum).


