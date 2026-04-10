---
title: " "
---

`easymode segment intermediate_filament`

The intermediate filament (IF) model outputs segmentations for the ~10 nm diameter filaments of intermediate thickness that are prevalent in many mammalian cell lines. Various different IFs exist, and when annotating tomograms at 10 A/px, we're not always sure which type we are looking at. This model has not yet been validated by STA. One common IF is vimentin, and we do believe that many of the filaments annotated as IFs in the training data are vimentin - the training dataset also includes some cell lines (for example, immortalized human skin fibroblasts) where vimentin is the dominant or only IF that is (thought to be) expressed.

**Example output**
<br>
<p style="text-align:center;">
  <video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:16/9; background:#fff; border-radius:8px; display:block; margin:auto;">
    <source src="../../assets/intermediate_filament.mp4" type="video/mp4">
    Video failed to load.
  </video>
</p>
Example of `easymode segment intermediate_filament` output overlaid on a tomogram of a human T lymphocyte (Jurkat). 





