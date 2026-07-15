---
title: easymode model library
---

This sections lists the features for which pretrained easymode models are currently available. We have split them up into three categories: 🐦 **species**, 🏞️ **biomes**, and ⚙️ **utility**. The distinction is purely descriptive.

The 🐦**species** include ribosomes, microtubules, actin, vault complexes, and other well-defined macromolecular assemblies that you might consider averaging. They are what you would call species in [M](https://github.com/warpem/warp) as well. 

The 🏞️ **biomes** category covers organelles and other cellular environments, such as mitochondria, the nuclear envelope, and the cytoplasm. These models can be used to sample the context within which the species are embedded; or in other words, the biomes are where the species live. 

The ⚙️ **utility** category currently covers two models: one for 'void', which maps what is and isn't sample and can be used to detect lamella boundaries, and one for ice particles.

You can always run `easymode list` to see the most up-to-date list of available models.

## Model gallery

<div style="display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:0.8em; margin:1.2em 0;">

<figure style="margin:0;"><a href="microtubule/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/microtubule.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="microtubule/">Microtubule</a></figcaption></figure>

<figure style="margin:0;"><a href="tric/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/tric.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="tric/">TRiC</a></figcaption></figure>

<figure style="margin:0;"><a href="ribosome/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/ribosome.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="ribosome/">Ribosome</a></figcaption></figure>

<figure style="margin:0;"><a href="membrane/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/membrane.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="membrane/">Membrane</a></figcaption></figure>

<figure style="margin:0;"><a href="actin/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/actin.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="actin/">Actin</a></figcaption></figure>

<figure style="margin:0;"><a href="intermediate_filament/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/intermediate_filament.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="intermediate_filament/">Intermediate filament</a></figcaption></figure>

<figure style="margin:0;"><a href="mitochondrion/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/mitochondrion.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="mitochondrion/">Mitochondrion</a></figcaption></figure>

<figure style="margin:0;"><a href="cytoplasm/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/cytoplasm.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="cytoplasm/">Cytoplasm</a></figcaption></figure>

<figure style="margin:0;"><a href="nuclear_envelope/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/nuclear_envelope.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="nuclear_envelope/">Nuclear envelope</a></figcaption></figure>

<figure style="margin:0;"><a href="npc/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/npc.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="npc/">Nuclear pore complex</a></figcaption></figure>

<figure style="margin:0;"><a href="impdh/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/impdh.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="impdh/">hfIMPDH</a></figcaption></figure>

<figure style="margin:0;"><a href="cytoplasmic_granule/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/cytoplasmic_granule.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="cytoplasmic_granule/">Cytoplasmic granule</a></figcaption></figure>

<figure style="margin:0;"><a href="mitochondrial_granule/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/mitochondrial_granule.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="mitochondrial_granule/">Mitochondrial granule</a></figcaption></figure>

<figure style="margin:0;"><a href="void/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/void.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="void/">Void</a></figcaption></figure>

<figure style="margin:0;"><a href="vault/"><video autoplay loop muted playsinline style="width:100%; aspect-ratio:16/9; background:#fff; border-radius:6px; display:block;"><source src="../assets/vault.mp4" type="video/mp4"></video></a><figcaption style="text-align:center; font-size:0.82em; margin-top:0.25em;"><a href="vault/">Vault</a></figcaption></figure>

</div>

## Model classification
We classify the current status of the models using three categories: 

1. Models marked with 🟢 are available for general use and have been validated in some way via subtomogram averaging. 
2. Models marked with 🔵 are available, and while we believe the output is useful for screening large datasets and 3D visualization, validation by subtomogram averaging is not (yet) done. 
3. Models marked with 🚧 are a work in progress; some are already online, but they remain experimental.

**Regardless of how accurate we think models may be, we always encourage you to try them out and inspect the results for yourself.**