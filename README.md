![easymode](assets/easymode_banner.png)

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

All networks were trained on a dataset of over 4500 tilt series from 68+ sources, covering many prokaryotic, archaeal, and eukaryotic species, different sample preparation techniques, hardware configurations, pixel sizes, electron doses, and defocus values. Tomograms were reconstructed in multiple flavours using [WarpTools](https://warpem.github.io/), [AreTomo3](https://github.com/czimaginginstitute/AreTomo3), [cryoCARE](https://github.com/juglab/cryoCARE_pip), and [DeepDeWedge](https://github.com/MLI-lab/DeepDeWedge) at 10.0 A voxel size. See the [preprint](https://www.biorxiv.org/content/10.64898/2026.05.19.726344v1) for more details.

## See our other tools

<p align="center">
  <a href="https://github.com/bionanopatterning/Ais"><img src="https://github.com/bionanopatterning/Ais/raw/master/docs/res/ais_banner.png" width="49%"></a>
  <a href="https://github.com/bionanopatterning/Pom"><img src="https://github.com/bionanopatterning/Pom/raw/main/docs/res/pom_banner.png" width="49%"></a>
</p>

Mart So-Last, 2026 | mgflast@gmail.com
