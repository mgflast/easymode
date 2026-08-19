---
title: " "
---

`easymode denoise --method n2n`

A [noise2noise](https://arxiv.org/abs/1803.04189) model trained on even/odd tomogram half-splits from 43 distinct sources. Thanks to two rounds of training (even/odd, raw/denoised), the final network can be applied directly to raw tomograms. Trained with Warp's tomogram reconstruction method.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1em;">
<video autoplay loop muted playsinline controls style="width:100%; background:#fff; border-radius:8px; display:block;">
  <source src="../../assets/denoise_raw.mp4" type="video/mp4">
  Video failed to load.
</video>
<video autoplay loop muted playsinline controls style="width:100%; background:#fff; border-radius:8px; display:block;">
  <source src="../../assets/denoise_n2n.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
A tomogram of a human T cell (Jurkat), raw input (left) and denoised output (right).
{: .subtitle }

See [general denoisers](../user_guide/functions/general_denoisers.md) for the full description and for the optional arguments to `easymode denoise`.
