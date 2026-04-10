---
title: " "
---

`easymode segment npc`

Unlike most species models, the npc (nuclear pore complex) model operates at 30 Å/px, resulting in ~27x faster inference. 

The model currently fails to accurately segment NPCs when they are approximately in-plane with the tomogram. Or in other words, when the nuclear envelope is approximately in-plane with the lamella.

We used dataset [EMPIAR-11943](https://www.ebi.ac.uk/empiar/EMPIAR-11943), consisting of 130 tilt series of FIB-milled D. discoideum cells in hyperosmotic stress, for validation of this model by subtomogram averaging.

**Example output**

<div style="display: flex; gap: 1em; flex-wrap: wrap;">
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline controls style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:8px;">
  <source src="../../assets/npc.mp4" type="video/mp4">
  Video failed to load.
</video>
<p>Example of <code>easymode segment npc</code> output overlaid on a tomogram from EMPIAR-11845 (FIB-milled <em>D. discoideum</em>).</p>
</div>
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline controls style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:8px;">
  <source src="../../assets/npc_map.mp4" type="video/mp4">
  Video failed to load.
</video>
<p>Subtomogram average obtained with <code>easymode segment npc</code> and <code>easymode pick npc</code> and averaging in RELION5/M.</p>
</div>
</div>

