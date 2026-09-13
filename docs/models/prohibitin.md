---
title: " "
---

`easymode segment prohibitin`

The prohibitin model is a shape-based prohibitin segmentation model and was trained on manually selected 3D subtomograms that were labelled using a 2D Ais UNet + post-processing. It was trained at the default easymode pixel size of 10 Å/px. 

**Example output**

<div style="display: flex; gap: 1em; flex-wrap: wrap;">
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline controls style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:8px;">
  <source src="../../assets/prohibitin.mp4" type="video/mp4">
  Video failed to load.
</video>
<p>Example of <code>easymode segment prohibitin</code> output overlaid on a tomogram of a human T lymphocyte (Jurkat).</p>
</div>
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline controls style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:8px;">
  <source src="../../assets/prohibitin_map.mp4" type="video/mp4">
  Video failed to load.
</video>
<p>Subtomogram average of prohibitin in <em>Polytomella spp.</em>, obtained with <code>easymode segment prohibitin</code> and <code>easymode pick prohibitin</code> and averaging in RELION5.</p>
</div>
</div>





