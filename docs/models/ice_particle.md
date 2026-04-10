---
title: " "
---

`easymode segment ice_particle`

This model segments ice particles, which are common contaminants on the top and bottom of cryoET samples. This 2D model works at 30 Å/px. It can be useful to segment ice particles when using template matching (TM) for particle detection: large dark blobs such as ice particles can often score high in TM, but are clearly false positives, so you can discard them by detecting what's ice and what isn't.


**Example output**
<br>
<p style="text-align:center;">
  <video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:16/9; background:#fff; border-radius:8px; display:block; margin:auto;">
    <source src="../../assets/ice_particles.mp4" type="video/mp4">
    Video failed to load.
  </video>
</p>
Example of `easymode segment ice_particle` output overlaid on a tomogram of a human (Jurkat) cell.
