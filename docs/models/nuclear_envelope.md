---
title: " "
---

`easymode segment nuclear_envelope`

The nuclear envelope (NE) model was trained at 30 Å/px and has a receptive field of almost 5000 Å. At this scale it processes tomograms of up to 768 x 768 x 384 nm in one go, usually within a few seconds.

This model does not currently label the NE when it is approximately in plane with the lamella. In these cases, the bounding membranes are not visible in the missing wedge-affected reconstructions, and we were not able to confidently annotate enough examples of this presentation to include it in the training collection. Using the nucleus and cytoplasm models and a number of datasets where we can now identify these orientations much better, we aim to resolve this issue in an upcoming iteration. 

<br>
<p style="text-align:center;">
  <video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:16/9; background:#fff; border-radius:8px; display:block; margin:auto;">
    <source src="../../assets/nuclear_envelope.mp4" type="video/mp4">
    Video failed to load.
  </video>
</p>
Example of `easymode segment nuclear_envelope` output overlaid on a tomogram from EMPIAR-11845 (FIB-milled D. discoideum).

