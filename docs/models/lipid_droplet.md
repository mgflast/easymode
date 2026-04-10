---
title: " "
---

`easymode segment lipid_droplet`

This model segments lipid droplets (LDs) and works at 50 Å/px. We haven't validated it very thoroughly; testing it against the training collection we find that it does not output many false positives on unseen tomograms, but since the number of lipid droplets in the collection is limited we haven't yet measured precision and recall on left-out instances of LDs. As with any other network, if you're interested in segmenting this feature, do a test run on a couple of your own tomograms and inspect the results to decide whether the network offers you the utility you'd need.


**Example output**
<br>
<p style="text-align:center;">
  <video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:16/9; background:#fff; border-radius:8px; display:block; margin:auto;">
    <source src="../../assets/lipid_droplet.mp4" type="video/mp4">
    Video failed to load.
  </video>
</p>
Example of `easymode segment ice_particle` output overlaid on a tomogram of a human (HeLa) cell (EMPIAR-13145).
