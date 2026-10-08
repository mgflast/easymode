---
title: " "
---

[MissAlignment](https://github.com/warpem/miss-alignment) (Chaillet et al., [bioRxiv 10.64898/2026.04.29.721716](https://doi.org/10.64898/2026.04.29.721716)) is a tilt series alignment tool that optimizes tilt series alignment by minimizing an alignment penalty that is calculated by a misalignment scoring function. This function is a trained network, and training can take quite a bit of time. See their publication for more information.

Using the easymode training data collection, we prepared a set of general and pretrained missalignment networks, trained using data from 54 separate training datasets. This included many different species, but also various different sample types and geometries including purified materials, cellular lamella, tissue lamella, whole cell periphery. In our preliminary and qualitative tests, alignments achieved with this general network set are no worse than those achieved with per-dataset bespoke networks.

[Download the models (27 MB)](https://huggingface.co/mgflast/easymode-v2/resolve/main/missalignment/easymode_missalignment.zip)

See [MissAlignment](../user_guide/functions/missalignment.md) for the full description and usage.
