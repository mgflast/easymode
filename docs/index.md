---
title: ""
hide:
  - toc
---

![easymode logo](assets/banner.png)

<h2 style="text-align: center; font-weight: bold;">Pretrained general networks for cellular cryoET</h2>
<h3 style="text-align: center; font-weight: bold; font-style: italic;">artisanally crafted in Cambridgeshire, UK</h3>

Trained on a large and curated body of cryoET datasets, **easymode** provides pretrained networks for feature detection in cellular cryoET. Our models are hosted via [Hugging Face](https://huggingface.co/mgflast/easymode/tree/main) and automatically distributed for inference – meaning you don't need to worry about downloading model weights.

Simply call `easymode segment ribosome`, point to your data, and easymode will do the rest. Or `microtubule`. Or `mitochondrion`. Or `npc`. Or `tric`. Or check out the [full model library](models/index.md).

## easymode ecosystem 
**easymode** was built on top of [Ais](https://github.com/bionanopatterning/Ais), inspired by [Membrain](https://github.com/CellArchLab/MemBrain-v2), and designed for use within [Warp](https://warpem.github.io)/[Relion](https://github.com/3dem/relion)/[M](https://warpem.github.io/) pipelines. By growing the easymode model library, our eventual goal is to enable [Pom's](https://github.com/bionanopatterning/Pom) tricks - context-aware particle picking, area selective template matching, and representing datasets as searchable databases - without requiring any training at all.
