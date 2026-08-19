---
title: " "
---

`easymode segment mip`

The microtubule inner protein (MIP) model segments densities found inside the microtubule lumen. These may be any sort of particles. Although some MIPs have been identified (cofilactin, for example: [Ventura Santos et al.](https://pubmed.ncbi.nlm.nih.gov/37702953/)) and some proteins have been suggested as MIPs ([Chakraborty et al.](https://www.pnas.org/doi/10.1073/pnas.2404017121)), the identity of most lumenal densities remains unknown. 

For validation: we used this model ourselves to pick and average MIPs in the dataset shown below. We find it works really well to pick lumenal densities and achieved a 3.1 Å resolution map of one particular MIP with it. This result is pending review so we will include it on this page later (but feel free to contact me at mgflast@gmail.com). Overall, MIPs appear to be quite diverse and we also believe that the composition of the MIPs might vary between cell types. So for picking and averaging, you'll probably need to do a lot of classification.

**Example output**
<br>
<p style="text-align:center;">
  <video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:16/9; background:#fff; border-radius:8px; display:block; margin:auto;">
    <source src="../../assets/mip.mp4" type="video/mp4">
    Video failed to load.
  </video>
</p>
Example of `easymode segment mip` output overlaid on a tomogram of human iPSC-derived neurons prepared as 'ghost cells' (surfactant treated, washing away the soluble components and leaving the insoluble fraction in place).
