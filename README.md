[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/mgflast/easymode/LICENSE.txt)
[![Downloads](https://img.shields.io/pypi/dm/easymode)](https://pypi.org/project/easymode/)
[![Documentation Status](https://img.shields.io/website?url=https%3A%2F%2Fmgflast.github.io%2Feasymode)](https://mgflast.github.io/easymode)
![Last Commit](https://img.shields.io/github/last-commit/mgflast/easymode)

## easymode
### pretrained neural networks for cellular cryoET

Easymode is a collection of general pretrained neural networks for cellular cryo-electron tomography (cryoET). It provides single command line interface functions to handle feature detection: segmentation of cellular features, coordinate extraction, and data preprocessing. Inspired by, built on, and built to work with: [Warp/M](https://github.com/warpem/warp), [RELION](https://github.com/3dem/relion), [MemBrain-seg](https://github.com/teamtomo/membrain-seg), [Ais](https://github.com/bionanopatterning/Ais), and [Pom](https://github.com/bionanopatterning/Pom).

**[Preprint](https://www.biorxiv.org/content/10.64898/2026.05.19.726344v1) | [Documentation](https://mgflast.github.io/easymode) | [Model library](https://mgflast.github.io/easymode/models/) | [Tutorials](https://mgflast.github.io/easymode/user_guide/overview/)**

### installation

```bash
conda create -n easymode python=3.10 cudatoolkit=11.2 cudnn=8.1 git -c conda-forge
conda activate easymode
pip install tensorflow==2.11.0
pip install git+https://github.com/mgflast/easymode.git
```

For full installation instructions including CUDA setup, Ais, and Pom, see the [installation guide](https://mgflast.github.io/easymode/user_guide/functions/installation/).

### quick start

```bash
# List available models
easymode list

# Segment features
easymode segment ribosome microtubule --data warp_tiltseries/reconstruction

# Extract particle coordinates
easymode pick ribosome --data segmented/ --output coordinates/ribosome --size 2000000 --spacing 300

# Reconstruct tomograms (requires WarpTools and AreTomo3)
easymode reconstruct --frames frames/ --mdocs mdocs/ --apix 1.56 --dose 4.6

# Denoise tomograms
easymode denoise --method n2n --mode direct --data warp_tiltseries/reconstruction --output denoised
```

Pretrained networks are hosted on [HuggingFace](https://huggingface.co/mgflast/easymode) and downloaded automatically on first use. See the [user guide](https://mgflast.github.io/easymode/user_guide/overview/) for detailed usage.

### training data

All networks were trained on a dataset of over 4000 tilt series from 50+ sources, covering many prokaryotic, archaeal, and eukaryotic species, different sample preparation techniques, hardware configurations, pixel sizes, electron doses, and defocus values. Tomograms were reconstructed in multiple flavours using [WarpTools](https://warpem.github.io/), [AreTomo3](https://github.com/czimaginginstitute/AreTomo3), [cryoCARE](https://github.com/juglab/cryoCARE_pip), and [DeepDeWedge](https://github.com/MLI-lab/DeepDeWedge) at 10.0 A voxel size. See the [preprint](https://www.biorxiv.org/content/10.64898/2026.05.19.726344v1) for more details.

### references

1. Tegunov et al., 'Multi-particle cryo-EM refinement with M visualizes ribosome-antibiotic complex at 3.5 A in cells', Nature Methods (2021): https://doi.org/10.1038/s41592-020-01054-7
2. Tegunov & Cramer, 'Real-time cryo-electron microscopy data preprocessing with Warp', Nature Methods (2019): https://doi.org/10.1038/s41592-019-0580-y
3. Peck et al., 'AreTomoLive: Automated reconstruction of comprehensively-corrected and denoised cryo-electron tomograms in real-time and at high throughput', bioRxiv (2025): https://doi.org/10.1101/2025.03.11.642690
4. Lamm et al., 'MemBrain v2: an end-to-end tool for the analysis of membranes in cryo-electron tomography', bioRxiv (2024): https://doi.org/10.1101/2024.01.05.574336
5. Burt et al., 'An image processing pipeline for electron cryo-tomography in RELION-5', FEBS Open Bio (2024): https://doi.org/10.1002/2211-5463.13873
6. Buchholz et al., 'Content-aware image restoration for electron microscopy', IEEE (2019): https://doi.org/10.1109/ISBI.2019.8759519
7. Wiedemann & Heckel, 'A deep learning method for simultaneous denoising and missing wedge reconstruction in cryogenic electron tomography', Nature Communications (2024): https://doi.org/10.1038/s41467-024-51438-y
8. Last et al., 'Streamlining segmentation of cryo-electron tomography datasets with Ais', eLife (2024): https://doi.org/10.7554/eLife.98552.3
