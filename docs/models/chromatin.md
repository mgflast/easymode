---
title: " "
---

`easymode segment chromatin`

Rather than a network that directly picks nucleosomes, we decided to include a network that segments chromatin as a whole. For training we used a mixture data from lamella (D. discoideum, H. sapiens neutrophils, H. sapiens T cells, M. musculus embryonic stem cells, H. sapiens HeLa cells, M. musculus metaphase cells), chromatin seen in semi-purified samples (H. sapiens mitotic spinle, S. uvarum mitotic spindle, H. sapiens iNeuron ghost cells), and tomograms of purified chromatin (CEMOVIS samples of chromatin, purified human chromatin). In the best of these tomograms we could clearly see individual nucleosomes, but in most cases it was difficult to unambiguously identify all individual nucleosomes within any field of view. 

If you want to use this model for nucleosome STA, you will need to do some classification. We include a [tutorial](../user_guide/examples/nucleosome.md) to demonstrate this.

The network was tested on a [dataset of human T cells](https://www.ebi.ac.uk/empiar/EMPIAR-13566/). This dataset one of the  very best for picking and averaging nucleosomes that we have seen, and illustrates an important aspect of using easymode and doing (high-resolution) STA: the main determinant of success in processing remains the quality of the input data. We also tested the easymode network on two small collections of tomograms from other sources (S. cerevisae semi-purified chromatin, H. sapiens HeLa nuclear periphery) where the overall segmentation output of chromatin was acceptable, but we still did not manage to average a nucleosome below 10.0 Å. 


**Example output**
<br>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1em;">
<video autoplay loop muted playsinline controls style="width:100%; background:#fff; border-radius:8px; display:block;">
  <source src="../../assets/chromatin.mp4" type="video/mp4">
  Video failed to load.
</video>
<video autoplay loop muted playsinline controls style="width:100%; background:#fff; border-radius:8px; display:block;">
  <source src="../../assets/nucleosome_average.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
Left: example of `easymode segment chromatin` output overlaid on a tomogram of a human primary T cell. The tomogram was denoised using the pretrained IsoNet2-style denoiser. Right: a nucleosome subtomogram average from this same dataset ([EMPIAR-13566](https://www.ebi.ac.uk/empiar/EMPIAR-13566/)) at 7.4 Å resolution.
{: .subtitle }
