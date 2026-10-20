# Pom installation

Pom is a companion package to easymode. Once you've segmented many different things with easymode (or [Ais](https://github.com/mgflast/Ais), or any other segmentation tool), you can use Pom to organise and curate the data.

Pom is installed together with easymode (`pip install easymode`), so there is nothing extra to install. To install Pom on its own, use `pip install pom-cryoet`.

## Pom browser app

The browser app (`pom browse`) needs streamlit, which is not installed by default because it conflicts with the protobuf requirement of tensorflow 2.11. To use the app, install streamlit:

```
pip install streamlit==1.56.0 streamlit-aggrid
```

With TensorFlow 2.15 or newer, that is all. With TensorFlow 2.11, then also install protobuf:

```
pip install protobuf==3.20.0
```

pip will warn that tensorflow or streamlit is incompatible with this protobuf version. Ignore that; it works. Do not do this with TensorFlow 2.20 or newer, which needs protobuf 5.28 or newer and does not import with 3.20.

## 3D visualization

Pom uses OpenGL and glfw for 3D visualization. The `pom render` command can only be run on a system that supports these, so headless servers can cause issues. On our cluster it works fine as long as we connect with X11 forwarding: `ssh -X user@entrypoint`, or `srun --x11 --pty -p agpu --gres=gpu:4 bash` when connecting to a GPU node.
