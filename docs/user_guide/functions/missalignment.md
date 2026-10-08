# MissAlignment

[MissAlignment](https://github.com/warpem/miss-alignment) (Chaillet et al., [bioRxiv 10.64898/2026.04.29.721716](https://doi.org/10.64898/2026.04.29.721716)) is a tilt series alignment tool that optimizes tilt series alignment by minimizing an alignment penalty that is calculated by a misalignment scoring function. This function is a trained network, and training can take quite a bit of time. See their publication for more information.

Using the easymode training data collection, we prepared a set of general and pretrained missalignment networks, trained using data from 54 separate training datasets. This included many different species, but also various different sample types and geometries including purified materials, cellular lamella, tissue lamella, whole cell periphery. In our preliminary and qualitative tests, alignments achieved with this general network set are no worse than those achieved with per-dataset bespoke networks.

## Usage

We aim for these networks to be included in MissAlignment itself, but in the meanwhile make them available here. To use them, [download](https://huggingface.co/mgflast/easymode-v2/resolve/main/missalignment/easymode_missalignment.zip) the models and unzip the archive anywhere. The resulting `easymode_missalignment/` folder holds `iter1/` to `iter8/`, each with a `model.ckpt`, in the same layout MissAlignment writes during training, plus an `inference_config.yaml` with our 8-iteration schedule. In that config, set `data_directory` to your Warp `warp_tiltseries/` directory and `model_run_directory` to the unzipped `easymode_missalignment/` folder. Then run:

```
miss-alignment infer --config-file /path/to/easymode_missalignment/inference_config.yaml --prepare-stacks 10.0
```

We refer to the [MissAlignment docs](https://github.com/warpem/miss-alignment/blob/main/docs/usage.md) for full details on the inference tools.

When using these networks, you can cite easymode if you find the general network has helped you, but cite MissAlignment first of all: we only did the pretraining.
