---
title: " "
---

`easymode segment ferritin`

This model for (apo)ferritin segmentation is a work in progress and only the preliminary 2D network is available at this time. We observed (apo)ferritin-like particles in multiple different data sources including some cryoFIB milled cells, where the particles are often found in what we think are endosomes. 

We're not really sure the current training labels are specific to (apo)ferritin, but at least the current network seems to pick very well in a dataset of purified apoferritin. If you're following the [Warp/M subtomogram averaging tutorial](https://warpem.github.io/warp/user_guide/warptools/quick_start_warptools_tilt_series/#template-matching), you could use this network as an alternative to template matching.

This page will be updated when the 3D network is ready and validation results (by subtomogram averaging) are available.

**Example output**

<div style="display: flex; gap: 1em; flex-wrap: wrap;">
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline controls style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:8px;">
  <source src="../../assets/ferritin.mp4" type="video/mp4">
  Video failed to load.
</video>
<p>Example of <code>easymode segment ferritin</code> output overlaid on a tomogram of a human (HeLa) cell.</p>
</div>
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline controls style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:8px;">
  <source src="../../assets/ferritin2.mp4" type="video/mp4">
  Video failed to load.
</video>
<p>Example of <code>easymode segment ferritin</code> output overlaid on a tomogram of purified apoferritin (EMPIAR-10491).</p>
</div>
</div>