# easymode + Ais & Pom

**easymode**, **Ais**, and **Pom** are related tools that together, in our experience, are very useful for exploring data, working with large cryoET datasets, and visualizing, organising, curating your cryoET project. 

## easymode

easymode provides pretrained general networks for common features in cellular cryoET. It is not the intended use case for users to train their own easymode networks — although it is possible. The main challenge in cryoET segmentation is not the machine learning itself, but the preparation of training labels. In cryoET we never have ground truth (except with simulated data), so preparing training labels is careful manual work. In creating easymode, we drew on our experience with Ais and Pom to do this at a reasonable scale.

If you are interested in segmenting your own feature of interest, you have a few options:

- **Let us know** what you would like a model for and consider submitting example tomograms using the `easymode report` function.
- **Before training a 3D network**, consider using [Ais](https://github.com/bionanopatterning/Ais) to train a 2D or 2.5D network — this is often easier and faster.

## Ais

[Ais](https://github.com/bionanopatterning/Ais) is specifically designed to streamline the preparation of training labels and the training of segmentation networks. The workflow is iterative: you annotate a bit, train, check what works and what doesn't, and then prepare additional annotations as needed. Because it is much easier and faster to annotate in 2D, we recommend trying Ais first for features that can reliably be identified in single tomogram slices.

- [Ais paper](https://elifesciences.org/reviewed-preprints/98552)
- [Ais documentation](https://ais-cryoet.readthedocs.io/en/latest/)
- [Ais on GitHub](https://github.com/bionanopatterning/Ais)

## Pom

[Pom](https://github.com/bionanopatterning/Pom) occupies a different place in the cryoET pipeline. The idea behind Pom ([paper](https://www.biorxiv.org/content/10.1101/2025.01.16.633326v1)) was that if we could annotate all the major cellular features, we could use this to visualize and summarize tomogram composition, and then represent a large dataset as a searchable database — rather than just a long list of files on disk. As datasets in cryoET are getting bigger and bigger, we think such approaches are critical.

- [Pom paper](https://www.biorxiv.org/content/10.1101/2025.01.16.633326v1)
- [Pom on GitHub](https://github.com/bionanopatterning/Pom)

At the time of its initial release, Pom also contained features for segmentation. We have since divided the work:

- **easymode** — pretrained networks, common features only.
- **Ais** — traing your own networks, when easymode does not provide.
- **Pom** — after segmentation, use Pom to visualize, browse, and curate your dataset, and when doing subtomogram averaging, to take measurements of particle environments.

