---
title: easymode model library
---

This sections lists the features for which pretrained easymode models are currently available. Most of them are segmentation models, which we have split up into three categories: 🐦 **species**, 🏞️ **biomes**, and ⚙️ **utility**. That distinction is purely descriptive. The fourth category, 🪄 **preprocessing**, is different in kind: it holds the general denoisers, which are applied to tomograms before segmentation rather than segmenting anything themselves.

The 🐦**species** include ribosomes, microtubules, actin, vault complexes, and other well-defined macromolecular assemblies that you might consider averaging. They are what you would call species in [M](https://github.com/warpem/warp) as well. 

The 🏞️ **biomes** category covers organelles and other cellular environments, such as mitochondria, the nuclear envelope, and the cytoplasm. These models can be used to sample the context within which the species are embedded; or in other words, the biomes are where the species live. 

The ⚙️ **utility** category currently covers two models: one for 'void', which maps what is and isn't sample and can be used to detect lamella boundaries, and one for ice particles.

The 🪄 **preprocessing** category covers the general denoisers, which are invoked via `easymode denoise --method` rather than `easymode segment`. Since there is nothing to average, their status markers indicate general availability rather than validation by subtomogram averaging.

You can always run `easymode list` to see the most up-to-date list of available models.

## Model classification
We classify the current status of the models using three categories: 

1. Models marked with 🟢 are available for general use and have been validated in some way via subtomogram averaging. 
2. Models marked with 🔵 are available, and while we believe the output is useful for screening large datasets and 3D visualization, validation by subtomogram averaging is not (yet) done. 
3. Models marked with 🚧 are a work in progress; some are already online, but they remain experimental.

**Regardless of how accurate we think models may be, we always encourage you to try them out and inspect the results for yourself.**