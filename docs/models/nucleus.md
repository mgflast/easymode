---
title: " "
---

`easymode segment nucleus`

The nucleus envelope model was trained at 50 Å/px and has a receptive field of 6400 Å. 

Segmenting the nucleus can be useful to generate masks for picking, or to add context to picked particles. For example, for subtomogram averaging (STA) of nuclear pore complexes (NPCs), it is important to assign a consistent polarity to every particle - i.e., to identify the distinct cytoplasmic and nuclear sides. During validation of the NPC model by STA, after generating an initial average with Relion5, we used the nucleus and cytoplasm models to refine the orientation of each individual particle.

<br>
<p style="text-align:center;">
  <video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:16/9; background:#fff; border-radius:8px; display:block; margin:auto;">
    <source src="../../assets/nucleus.mp4" type="video/mp4">
    Video failed to load.
  </video>
</p>
Example of `easymode segment nucleus --2d` output overlaid on a tomogram from EMPIAR-11845 (FIB-milled D. discoideum) - note that while this dataset was included in the training collection, the tomogram was not.

