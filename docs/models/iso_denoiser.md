---
title: " "
---

`easymode denoise --method iso`

A distilled [IsoNet2](https://doi.org/10.1038/s41467-022-33957-8) network. Separate IsoNet2 teacher networks were trained on each of 48 half-split datasets and then distilled into a single student network, so that the final network is general and can be applied directly to raw tomograms.  Trained with Warp's tomogram reconstruction method.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1em;">
<video autoplay loop muted playsinline controls style="width:100%; background:#fff; border-radius:8px; display:block;">
  <source src="../../assets/denoise_raw.mp4" type="video/mp4">
  Video failed to load.
</video>
<video autoplay loop muted playsinline controls style="width:100%; background:#fff; border-radius:8px; display:block;">
  <source src="../../assets/denoise_iso.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
A tomogram of a human T cell (Jurkat), raw input (left) and denoised output (right).
{: .subtitle }

See [general denoisers](../user_guide/functions/general_denoisers.md) for the full description and for the optional arguments to `easymode denoise`.
