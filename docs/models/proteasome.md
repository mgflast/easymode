---
title: " "
---

`easymode segment proteasome`

This model for proteasome segmentation is a work in progress and only the preliminary 2D network is available at this time. The outputs seems qualitatively decent on a couple of test datasets, albeit with some slight erroneous activation on segments of filaments that have approximately the same diameter as proteasomes. 

This page will be updated when the 3D network is ready and validation results (by subtomogram averaging) are available.


**Example output**

<div style="display: flex; gap: 1em; flex-wrap: wrap;">
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline controls style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:8px;">
  <source src="../../assets/proteasome.mp4" type="video/mp4">
  Video failed to load.
</video>
<p>Example of <code>easymode segment proteasome</code> output overlaid on a tomogram of a D. discoideum cell (EMPIAR-13145).</p>
</div>
</div>