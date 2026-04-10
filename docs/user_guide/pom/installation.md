# Pom installation

Pom is a companion package to easymode. Once you've segmented many different things with easymode (or [Ais](https://github.com/bionanopatterning/Ais), or any other segmentation tool), you can use Pom to organise and curate the data.

Pom can be installed in the same environment as easymode, but if you are using easymode with a tensorflow backend there is a slight issue with dependencies. Just follow these instructions and ignore the errors which pip initially gives:

```
conda activate easymode
pip install git+https://github.com/bionanopatterning/Pom.git
```

Then fix the dependency problem by manually installing `protobuf==3.20.3`. Either tensorflow or streamlit will complain that this is incompatible, but that isn't true.

```
pip install protobuf==3.20.3
```

**Alternatively, install Pom in a separate environment**:

```
conda create -n pom python=3.10
conda activate pom
pip install git+https://github.com/bionanopatterning/Pom.git
```

## 3D visualization

Pom uses OpenGL and glfw for 3D visualization. The `pom render` command can only be run on a system that supports these, so headless servers can cause issues. On our cluster it works fine as long as we connect with X11 forwarding: `ssh -X user@entrypoint`, or `srun --x11 --pty -p agpu --gres=gpu:4 bash` when connecting to a GPU node.
