---
title: easymode model library
---

This sections lists the features for which pretrained easymode models are currently available. We have split them up into three categories: 🐦 **species**, 🏞️ **biomes**, and ⚙️ **utility**. The distinction is purely descriptive.

The 🐦**species** include ribosomes, microtubules, actin, vault complexes, and other well-defined macromolecular assemblies that you might consider averaging. They are what you would call species in [M](https://github.com/warpem/warp) as well. 

The 🏞️ **biomes** category covers organelles and other cellular environments, such as mitochondria, the nuclear envelope, and the cytoplasm. These models can be used to sample the context within which the species are embedded; or in other words, the biomes are where the species live. 

The ⚙️ **utility** category currently covers two models: one for 'void', which maps what is and isn't sample and can be used to detect lamella boundaries, and one for ice particles.

You can always run `easymode list` to see the most up-to-date list of available models.

## Model classification
We classify the current status of the models using three categories: 

1. Models marked with 🟢 are available for general use and have been validated in some way via subtomogram averaging. 
2. Models marked with 🔵 are available, and while we believe the output is useful for screening large datasets and 3D visualization, validation by subtomogram averaging is not (yet) done. 
3. Models marked with 🚧 are a work in progress; some are already online, but they remain experimental.

**Regardless of how accurate we think models may be, we always encourage you to try them out and inspect the results for yourself.**