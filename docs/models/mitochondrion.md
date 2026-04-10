---
title: " "
---

`easymode segment mitochondrion`

The mitochondrion model was trained at 50 Å/px and has a receptive field of 6400 Å. As a result, it is mostly able to recognize the full extent of mitochondria, and at high magnifications can be applied to tomograms in one go - no sliding window required, and thus utilizing the global context of the tomogram. The model was trained on manually curated 2D Ais UNet generated pseudolabels, using training data from human, mouse, baker's yeast, fission yeast, chlamydomonas, dictyostelium discoideum, and a number of other eukaryotic species, as well as on various prokaryotic species to include counterexamples to mitochondria. 

**Example output**
<br>
<p style="text-align:center;">
  <video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:16/9; background:#fff; border-radius:8px; display:block; margin:auto;">
    <source src="../../assets/mitochondrion.mp4" type="video/mp4">
    Video failed to load.
  </video>
</p>
Example of `easymode segment mitochondrion` output overlaid on a tomogram of a human (HeLa) cell.

