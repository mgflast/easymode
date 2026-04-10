---
title: " "
---

`easymode segment hfimpdh`

This model segments (human) IMPDH filaments (inosine monophosphate dehydrogenase - see [Johnson & Kollman, eLife (2020)](https://elifesciences.org/articles/53243) for an introduction). We found instances of these filaments in 8 different datasets of the [training data collection](../training_collection.md), including in tomograms from [EMPIAR-13145](https://www.ebi.ac.uk/empiar/EMPIAR-13145/) (in more than 10 tomograms), [EMPIAR-10989](https://www.ebi.ac.uk/empiar/EMPIAR-10989/) (1 tomogram), [EMPIAR-11538](https://www.ebi.ac.uk/empiar/EMPIAR-11538/) (>20 tomograms), [EMPIAR-12460](https://www.ebi.ac.uk/empiar/EMPIAR-12460/) (2 tomograms, mouse embryonic stem cells), [EMPIAR-11561](https://www.ebi.ac.uk/empiar/EMPIAR-11561/) (2 tomograms), EXT-007 (human U2OS cells, 2 tomograms), and LMB-014 (human Jurkat cells, 4 tomograms). While going through the routine of annotating in 2D, training a network, and running inference on the entire easymode training collection (~4300 tomograms at that point), we kept finding more of these. The filaments are relatively rare but turned out more common than we originally anticipated.

All but one of these datasets ([EMPIAR-12460](https://www.ebi.ac.uk/empiar/EMPIAR-12460/)) were of cultured human cells, so we do not yet know how well the network generalizes to other species. Moreover, we annotated IMPDH only in the filamentous state. This is why we named the model `hfimpdh`: **h**uman **f**ilamentous **IMPDH**.

We tested the network output qualitatively via inference on the complete training data collection and inspecting the results in Pom (as was done for all other networks), and for utility in subtomogram averaging by applying it to our own dataset of ~1000 mycophenolic acid (MPA) treated HeLa cell tomograms. MPA is an IMPDH inhibitor whose treatment leads to filament and bundle formation. Using a subset of 19 tomograms that contained IMPDH filaments and globular picking in easymode (spacing coordinates at 110 Å within the segmented filament regions - we did not use filament tracing because in the tightly packed bundles the segmentations touch, which makes filament tracing unreliable), we picked ~10,000 particles which with Relion5 and M and using D4 symmetry averaged to 4.0 Å resolution without requiring classification.

IMPDH filaments may be a somewhat niche target compared to the other networks in the library; we included it because we study IMPDH filament bundles using cryoET ourselves and may be a bit hyperfocused there. Please let us know if you find them in your data? 🙃

**Example output**

<div style="display: flex; gap: 1em; flex-wrap: wrap;">
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline controls style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:8px;">
  <source src="../../assets/impdh.mp4" type="video/mp4">
  Video failed to load.
</video>
<p>Example of <code>easymode segment hfimpdh --2d</code> output overlaid on a tomogram of a human (HeLa) cell treated with IMPDH inhibitor <em>mycophenolic acid</em>, leading to filament and bundle formation.</p>
</div>
<div style="flex: 1; min-width: 300px;">
<video autoplay loop muted playsinline controls style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:8px;">
  <source src="../../assets/impdh_map.mp4" type="video/mp4">
  Video failed to load.
</video>
<p>Composite map of an IMPDH filament, obtained with <code>easymode segment hfimpdh</code> and <code>easymode pick</code> and averaging in Relion5/M.</p>
</div>
</div>
