---
title: " "
---

`easymode segment actin`

The actin model was trained to output a cylindrical tube with a diameter of 60 Å along actin filaments. It was one of the most challenging models to train thus far and remains somewhat experimental.

In combination with the `--filament` flag in `easymode pick`, the model enables tracing actin filaments and picking of particles at regular intervals along individual filaments, with an **accurate prior on the particle orientation** and class labels linking particles to the parent filaments (_aisFilamentID). Use the `--per-filament` flag to write .star files for individual actin filaments. 

This model does not currently label cross-sections of actin, as seen when the filaments are approximately parallel to the Z axis of the tomogram, as we were unable to confidently annotated enough examples of such views. When working with data from cells cultured on grid, this limitation may be less of an issue, as actin filaments are often oriented parallel to the grid plane. However, when working with cells deposited onto a grid just prior to plunging or with lift-out data, it is something to keep in mind.

**Example output**

<div style="display: flex; gap: 1em; flex-wrap: wrap;">
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline controls style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:8px;">
  <source src="../../assets/actin.mp4" type="video/mp4">
  Video failed to load.
</video>
<p>Example of <code>easymode segment actin</code> output overlaid on a tomogram of human immortalized fibroblast 'ghost cells' (plasma membrane removed by surfactant treatment).</p>
</div>
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline controls style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:8px;">
  <source src="../../assets/actin_map.mp4" type="video/mp4">
  Video failed to load.
</video>
<p>Subtomogram average at 3.9 Å global resolution, obtained with <code>easymode segment actin</code> and <code>easymode pick actin</code> and averaging in RELION5/M.</p>
</div>
</div>


