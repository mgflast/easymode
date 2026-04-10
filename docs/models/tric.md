---
title: " "
---

`easymode segment tric`

The tric (T-complex protein Ring Complex) model was trained to output 170 Å spheres in the position of TRiC complexes. As in most other cases, the training labels were generated with an Ais 2D UNet applied to manually selected 3D subtomograms with a high degree of test-time augmentation. The resulting labels were post-processed to draw spherical labels centred on the TRiC particles.

For validation we used a set of 360 tomograms from HEK293T cells, the untreated condition of dataset EMPIAR-11538. After segmentation with easymode and picking globular particles with Ais, we used a set of 3202 particles for subtomogram averaging. One round of 3D classification yielded a subset of 620 particles in the open confirmation (see EMD-18922), alongside two classes of 943 and 52 particles in the closed state (see EMD-18927) and a remaining class of 1587 particles which did not appear to represent the TRiC complex (even though visual inspection of these particles in the original tomograms did suggest that a substantial fraction of the particles were genuine TRiC complexes). Refinement of the 943 closed-state particles with D8 symmetry yielded a map at 7.9 Å resolution.


**Example output**

<div style="display: flex; gap: 1em; flex-wrap: wrap;">
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline controls style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:8px;">
  <source src="../../assets/tric.mp4" type="video/mp4">
  Video failed to load.
</video>
<p>Example of <code>easymode segment tric</code> output overlaid on a tomogram from EMPIAR-11538 (FIB-milled <em>D. discoideum</em>).</p>
</div>
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline controls style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:8px;">
  <source src="../../assets/tric_map.mp4" type="video/mp4">
  Video failed to load.
</video>
<p>Subtomogram average at 7.9 Å resolution (D8 symmetry, 943 particles), obtained with <code>easymode segment tric</code> and <code>easymode pick tric</code> and averaging in RELION5/M.</p>
</div>
</div>