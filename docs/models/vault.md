---
title: " "
---

`easymode segment vault`

The vault model was trained on manually curated 3D subtomograms that were labelled by a 2D Ais UNet. The Ais net output a shape-based segmentation of vaults; as a result, the 3D easymode network also outputs a vault shape, although with strong missing wedge artefacts. This does not really matter for picking.

For validation we used dataset [EMPIAR-11845](https://www.ebi.ac.uk/empiar/EMPIAR-11845), consisting of 152 tomograms of FIB-milled D. discoideum cells. For the sake of the validation we re-trained the vault networks with data from this source excluded from the training collection.  

Vaults remain rare even in this dataset; we found on average between 2 to 3 particles per tomogram, or 393 in total. With D39 symmetry, subtomogram averaging plateaued at a resolution of 13.8 Å. 

<p style="text-align:center;">
  <video autoplay loop muted playsinline controls style="width:50%; max-width:600px; border-radius:8px; display:block; margin:auto;">
    <source src="../../assets/vault_map.mp4" type="video/mp4">
    Video failed to load.
  </video>
  Subtomogram average of the <em>D. discoideum</em> vault complex at 13.8 Å resolution with D39 symmetry.
</p>

**Example output**
<br>
<p style="text-align:center;">
  <video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:16/9; background:#fff; border-radius:8px; display:block; margin:auto;">
    <source src="../../assets/vault.mp4" type="video/mp4">
    Video failed to load.
  </video>
</p>
Example of `easymode segment vault` output overlaid on a tomogram from EMPIAR-11899 (FIB-milled D. discoideum).





