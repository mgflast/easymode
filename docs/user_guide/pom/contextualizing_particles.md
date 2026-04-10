# Contextualizing particles

`pom contextualize` combines segmentation results and particle coordinates (star files) to measure each particle's relationship to surrounding cellular features. If you've used `pom add_source` to register your tomograms and segmentations, this function can take these measurements automatically.

**Arguments:**

- `--starfile` — path to the particle `.star` file defining the coordinates/poses around which to measure.
- `--tomo-column` — the column in the star file that identifies the tomogram to which each particle belongs. Defaults to `rlnMicrographName` or `wrpSourceName`, so within a Warp/Relion/M pipeline you typically don't need to set this.
- `--substitutions` — substitutions (in `search:replace` format) to apply to micrograph names when matching to segmentation files. See [file name conventions](data_browser.md#sources-and-file-name-conventions).
- `--out_star` — path to the output `.star` file.
- `--apix` — pixel size of the coordinates in the star file.
- `--samplers` — the measurements Pom will take. These are explained below.

## Samplers

`pom contextualize` can perform two types of measurements: **volumetric**, where we measure the average segmentation value within some radius around a particle, and **distance**, where we measure the distance to the nearest segmented object.

### Volumetric samplers

Defined with two terms: `feature:radius`. This measures the average segmentation value within a sphere of the given radius (in Angstrom) around each particle. For example, `mitochondrion:300` measures how much of the volume within 300 Å of each particle is mitochondrion. Results are saved to a new column called `pomFeatureRadiusA` (e.g. `pomMitochondrion300A`) with values in the range 0.0–1.0.

### Distance samplers

Defined with three terms: `feature:threshold:dust`. Distance measurements require thresholding the segmentation output to binarize it. The `dust` parameter filters out small blobs that can arise from noisy segmentations: a positive value (e.g. `1e6`) ignores all blobs smaller than that volume in cubic Angstrom; a negative integer (e.g. `-1`) keeps only the N largest objects.

For example, `membrane:0.5:1e6` measures the distance to the nearest membrane, thresholded at 0.5, ignoring blobs smaller than 1,000,000 Å³. Results are saved to a column called `pomDistFeatureTThreshold` (e.g. `pomDistMembraneT50`) with values in Angstrom. Positive values mean the particle is outside the segmented region; negative values mean it is inside.

### Position offsets

For both sampler types, you can append a position offset: `:+distance` or `:-distance` (in Angstrom). The measurement is then taken not at the particle position itself, but at a point offset along the particle's primary axis (+Z in local space; the blue arrow in ArtiaX). This is useful for probing the environment in front of or behind a particle.

When using an offset, `pDistance` or `mDistance` is appended to the column name. For example, a volumetric sampler `cytoplasm:300:+100` produces column `pomCytoplasm300Ap100`, and a distance sampler `membrane:0.5:1e6:+100` produces column `pomDistMembraneT50p100`.

## Examples

**Classifying free vs. membrane-bound ribosomes**

Given `particles.star` (a post-refinement star file with ribosome coordinates) and membrane segmentations registered via `pom add_source`:

```
pom contextualize --starfile particles.star --apix 1.0 --substitutions .tomostar:_10.00Apx --samplers membrane:0.5:1e6 --out_star particles.star
```

**Determining the nuclear vs. cytoplasmic side of nuclear pore complexes** — see the [NPC orientation tutorial](../examples/npc_orientation.md).

**Measuring the distance to lamella surfaces** — see the [lamella surface distance tutorial](../examples/lamella_surface_distance.md).
