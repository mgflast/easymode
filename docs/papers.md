# Papers

## easymode ecosystem

**Ais** — Iterative 2D/2.5D segmentation training for cryoET. Ais provides the interactive annotation and training workflow that we used to prepare easymode's training labels at scale. If easymode doesn't cover your feature of interest, Ais is the tool to train your own network.

- Last et al. (2024) *Streamlining segmentation of cryo-electron tomography datasets with Ais* eLife. [Paper](https://elifesciences.org/articles/98552) · [GitHub](https://github.com/bionanopatterning/Ais)

**Pom** — Dataset-scale visualization for segmented cryoET data. After segmenting with easymode or Ais, Pom turns a large dataset into a searchable, browsable collection and helps with particle contextualization and filtering.

- Last et al. (2025). *Scaling data analyses in cellular cryoET using comprehensive segmentation* bioRxiv. [Paper](https://www.biorxiv.org/content/10.1101/2025.01.16.633326v1) · [GitHub](https://github.com/bionanopatterning/Pom)

## Related tools we like

**MemBrain-seg** — 3D membrane segmentation for cryoET, and the inspiration for easymode's network architecture.

- Lamm et al. (2024). *MemBrain v2: an end-to-end tool for the analysis of membranes in cryo-electron tomography.* bioRxiv. [Paper](https://www.biorxiv.org/content/10.1101/2024.01.05.574336v1) · [GitHub](https://github.com/teamtomo/membrain-seg)

**Warp / WarpTools** — On-the-fly processing for cryo-EM and cryoET, including motion correction, CTF estimation, and tomogram reconstruction. All easymode training data was processed using WarpTools.

- Tegunov & Cramer (2019). *Real-time cryo-electron microscopy data preprocessing with Warp.* Nature Methods. [Paper](https://doi.org/10.1038/s41592-019-0580-y) · [Docs](https://warpem.github.io/warp/)

**M** — Multi-particle refinement for cryo-EM and cryoET, used for easymode validation via subtomogram averaging.

- Tegunov et al. (2021). *Multi-particle cryo-EM refinement with M visualizes ribosome-antibiotic complex at 3.5 Å in cells.* Nature Methods. [Paper](https://doi.org/10.1038/s41592-020-01054-7) · [Docs](https://warpem.github.io/warp/)

**RELION** — Regularised likelihood optimisation for cryo-EM and cryoET. We used RELION5 for subtomogram averaging during easymode validation.

- Burt et al. (2024). *An image processing pipeline for electron cryo-tomography in RELION-5.* FEBS Open Bio. [Paper](https://doi.org/10.1002/2211-5463.13873) · [GitHub](https://github.com/3dem/relion)
