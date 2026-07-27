# Ais

**easymode** ships pretrained networks for common cellular features. When you need something that easymode does *not* cover, [Ais](https://mgflast.github.io/Ais/) is the tool for training your own segmentation network — it was designed to make annotation and training as fast and intuitive as possible, and easymode itself was built on top of it.

## When to reach for Ais

- **A feature easymode has no model for.** If the [model library](../../models/index.md) doesn't include your target, Ais lets you annotate a small amount of data and train a network for it.
- **Fast, iterative work in 2D.** Because it is much easier and faster to annotate in 2D than in 3D, we recommend trying Ais first for features that can reliably be identified in single tomogram slices. The workflow is iterative: annotate a bit, train, check what works, then add annotations where needed.
- **Manual picking and inspection.** Ais includes a built-in isosurface renderer and tools for particle picking and mesh extraction.

easymode's own `easymode pick` step [wraps Ais](../functions/picking.md) to turn segmented volumes into coordinate `.star` files, so you already have Ais installed as a dependency.

## Where to go next

The full Ais user guide — installation, annotation, training, batch processing, rendering, and the command line interface — lives on the dedicated Ais documentation site:

- 📖 **[Ais documentation](https://mgflast.github.io/Ais/)** — the complete user guide
- 🗄️ **[Ais model repository](https://www.aiscryoet.org)** — download and share trained models
- 💻 **[Ais on GitHub](https://github.com/mgflast/Ais)**
- 📜 **[Ais paper (eLife)](https://elifesciences.org/articles/98552)**

For a higher-level view of how easymode, Ais, and Pom fit together, see [easymode + Ais & Pom](../../ais_and_pom.md).
