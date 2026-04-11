[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/mgflast/easymode/LICENSE.txt)
[![Downloads](https://img.shields.io/pypi/dm/easymode)](https://pypi.org/project/easymode/)
[![Documentation Status](https://img.shields.io/website?url=https%3A%2F%2Fmgflast.github.io%2Feasymode)](https://mgflast.github.io/easymode)
![Last Commit](https://img.shields.io/github/last-commit/mgflast/easymode)

## easymode
### pretrained neural networks for cellular cryoET

Easymode is a collection of general pretrained neural networks for cellular cryo-electron tomography (cryoET). It provides single command line interface functions to handle feature detection: segmentation of cellular features, coordinate extraction, and data preprocessing. Inspired by and based upon [MemBrain-seg](https://github.com/teamtomo/membrain-seg).

**[Documentation](https://mgflast.github.io/easymode) | [Model library](https://mgflast.github.io/easymode/models/) | [Tutorials](https://mgflast.github.io/easymode/user_guide/overview/)**

### installation

```bash
conda create -n easymode python=3.10 cudatoolkit=11.2 cudnn=8.1 git -c conda-forge
conda activate easymode
pip install tensorflow==2.11.0 protobuf==3.20.3
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

All networks were trained on a dataset of over 2000 tilt series from 50+ sources, covering many prokaryotic, archaeal, and eukaryotic species, different sample preparation techniques, hardware configurations, pixel sizes, electron doses, and defocus values. Tomograms were reconstructed in multiple flavours using [WarpTools](https://warpem.github.io/), [AreTomo3](https://github.com/czimaginginstitute/AreTomo3), [cryoCARE](https://github.com/juglab/cryoCARE_pip), and [DeepDeWedge](https://github.com/MLI-lab/DeepDeWedge) at 10.0 A voxel size.

Not all training data sources are currently listed; a number of contributors prefer to remain anonymous so long as their work is not published. These contributions are marked with *.

| ID          | Contributor / source                                                 | Sample type                                | N (annotated) | Pixel size (A) |
|-------------|----------------------------------------------------------------------|--------------------------------------------|---------------|----------------|
| 001_HELA    | Mart Last                                                            | milled H. sapiens (HeLa)                   | 60 (59)       | 1.51           |
| 002_U2OS    | Mart Last                                                            | milled H. sapiens (U2OS)                   | 40 (26)       | 2.15           |
| 003_HSPERM  | Tom Dendooven, Alia dos Santos, Matteo Allegretti                    | milled H. sapiens (spermatozoa)            | 56 (41)       | 1.50           |
| 004_*       | *                                                                    | *                                          | 23 (20)       | 1.68           |
| 005_FIBRO   | Tom Hale                                                             | milled H. sapiens (fibroblasts)            | 52 (47)       | 1.33           |
| 006_*       | *                                                                    | *                                          | 20 (16)       | 1.69           |
| 007_APOF    | EMPIAR-10491                                                         | purified apoferritin                       | 37 (18)       | 0.79           |
| 008_HIV     | EMPIAR-10164                                                         | purified HIV particles                     | 10 (4)        | 0.68           |
| 009_SCEREV  | Sebastian Tacke, Elisa Lisicki, <br/>Tatjana Taubitz, Stefan Raunser | milled (hpf, pfib) S. cerevisiae           | 64 (51)       | 1.56           |
| 010_RIBO    | EMPIAR-11111                                                         | purified E. coli 70S ribosomes             | 25 (19)       | 1.07           |
| 011_CHLO    | EMPIAR-12612                                                         | milled S. oleracea chloroplasts            | 23 (18)       | 3.52           |
| 012_CHLAMY  | EMPIAR-11830                                                         | milled C. reinhardtii                      | 52 (50)       | 1.96           |
| 013_DIAT    | EMPIAR-11747                                                         | milled T. pseudonana                       | 7  (1)        | 1.07           |
| 014_CILIA   | EMPIAR-11078                                                         | milled C. reinhardtii ciliary base         | 23 (19)       | 3.42           |
| 015_MMVOLTA | CDPDS-10452                                                          | whole M. mycoides cells                    | 15 (15)       | 1.53           |
| 016_PHANTOM | CDPDS-10440, CDPDS-10445                                             | E. coli lysate with added proteins         | 19 (17)       | 1.53           |
| 017_MYCP    | EMPIAR-10499                                                         | whole M. pneunomiae cells                  | 65 (27)       | 1.70           |
| 018_ECM     | EMPIAR-11897                                                         | lift-out H. sapiens (extracellular matrix) | 39 (24)       | 2.14           |
| 019_ECOLI   | EMPIAR-12413                                                         | milled E. coli                             | 44 (19)       | 1.90           |
| 020_*       | *                                                                    | *                                          | 30 (25)       | 2.13           |
| 021_*       | *                                                                    | *                                          | 8  (7)        | 3.02           |
| 022_SCOV    | EMPIAR-10493                                                         | purified SARS-CoV-2 virions                | 20 (12)       | 1.53           |
| 023_SPORE   | EMPIAR-12176                                                         | milled E. intestinalis                     | 24 (11)       | 2.06           |
| 024_*       | *                                                                    | *                                          | 17 (6)        | 1.96           |
| 025_RPE     | EMPIAR-10989                                                         | cellular periphery H. sapiens (RPE1)       | 3  (3)        | 3.45           |
| 026_EHV     | EMPIAR-11896                                                         | Emiliania huxleyi virus 201                | 40 (10)       | 2.08           |
| 027_NUCFT   | Forson Gao                                                           | milled S. cerevisiae nuclei                | 21 (15)       | 1.51           |
| 028_ROOF    | CDPDS-10434                                                          | cellular periphery H. sapiens (HEK293)     | 20 (19)       | 2.17           |
| 029_TKIV    | EMPIAR-11058                                                         | milled T. kivui                            | 17 (7)        | 3.52           |
| 030_LDN     | Mart Last                                                            | cellular periphery H. sapiens (U2OS)       | 26 (7)        | 2.74           |
| 031_MITO    | Mart Last                                                            | milled H. sapiens (HeLa, mitochondria)     | 63 (59)       | 1.34           |
| 032_*       | *                                                                    | *                                          | 40 (24)       | 1.63           |
| 033_NPC     | EMPIAR-11830 (same source as 012_CHLAMY)                             | milled C. reinhardtii (nuclear envelope)   | 36 (36)       | 1.96           |
| 034_DICTYO  | EMPIAR-11845                                                         | milled D. discoideum                       | 152 (68)      | 2.18           |
| 035_GEM     | EMPIAR-11561                                                         | milled H. sapiens (HeLa, mitochondria)     | 15 (14)       | 3.43           |
| 036_MACRO   | EMPIAR-12457                                                         | milled H. sapiens (macrophages)            | 39 (21)       | 2.41           |
| 037_MESWT   | EMPIAR-12460                                                         | milled M. musculus (embryonic stem cell)   | 159 (26)      | 2.68           |
| 038_POMBE   | EMPIAR-10988                                                         | milled S. pombe                            | 9 (6)         | 3.37           |
| 039_JUMBO   | EMPIAR-11198                                                         | milled E. amylovora + RAY phage            | 32 (4)        | 4.27           |
| 040_SLO     | CDPDS-10004                                                          | milled (hpf, pfib) C. elegans              | 100 (24)      | 1.50           |
| 041_RPEM    | Cong Yu                                                              | milled H. sapiens (RPE1)                   | 17 (7)        | 1.57           |
| 042_NPCSC   | EMPIAR-10466                                                         | milled S. cerevisiae                       | 177 (0)       | 3.45           |
| 043_DICTY2  | EMPIAR-11899 (to be included after validation)                       | milled D. discoideum                       | 0 (0)         | 1.22           |
| 044_JURKAT  | Mart Last                                                            | milled H. sapiens (Jurkat)                 | 177 (0)       | 1.97           |
| 045_NPHL    | *                                                                    | *                                          | 231 (0)       | 1.56           |
| 046_ROOF2   | CDPDS-10431                                                          | cellular periphery H. sapiens (HEK293)     | 87 (0)        | 2.17           |
| 047_ECPP7   | CDPDS-10455                                                          | E. coli + PP7 virus-like particles         | 10 (0)        | 1.50           |
| 048_ELSO    | CDPDS-10444                                                          | human endo-/lysosomes                      | 10 (0)        | 1.54           |
| 049_CHR     | *                                                                    | *                                          | 46            | 1.97           |
| 050_MHSP    | EMPIAR-13145                                                         | milled H. sapiens (HeLa)                   | 239           | 2.31           |
| 051_ASSC    | *                                                                    | milled S. cerevisiae                       | 82            | 2.41           |

EMPIAR: [EM Public Image Archive](https://www.ebi.ac.uk/empiar/) | CDPDS: [CryoET Data Portal](https://cryoetdataportal.czscience.com/)

### references

1. Tegunov et al., 'Multi-particle cryo-EM refinement with M visualizes ribosome-antibiotic complex at 3.5 A in cells', Nature Methods (2021): https://doi.org/10.1038/s41592-020-01054-7
2. Tegunov & Cramer, 'Real-time cryo-electron microscopy data preprocessing with Warp', Nature Methods (2019): https://doi.org/10.1038/s41592-019-0580-y
3. Peck et al., 'AreTomoLive: Automated reconstruction of comprehensively-corrected and denoised cryo-electron tomograms in real-time and at high throughput', bioRxiv (2025): https://doi.org/10.1101/2025.03.11.642690
4. Lamm et al., 'MemBrain v2: an end-to-end tool for the analysis of membranes in cryo-electron tomography', bioRxiv (2024): https://doi.org/10.1101/2024.01.05.574336
5. Burt et al., 'An image processing pipeline for electron cryo-tomography in RELION-5', FEBS Open Bio (2024): https://doi.org/10.1002/2211-5463.13873
6. Buchholz et al., 'Content-aware image restoration for electron microscopy', IEEE (2019): https://doi.org/10.1109/ISBI.2019.8759519
7. Wiedemann & Heckel, 'A deep learning method for simultaneous denoising and missing wedge reconstruction in cryogenic electron tomography', Nature Communications (2024): https://doi.org/10.1038/s41467-024-51438-y
8. Last et al., 'Streamlining segmentation of cryo-electron tomography datasets with Ais', eLife (2024): https://doi.org/10.7554/eLife.98552.3
