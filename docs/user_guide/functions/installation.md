# Installation

## Installing from scratch
The following command will create a new environment called `easymode` and install easymode and all dependencies into it, including Ais and Pom. `tensorflow[and-cuda]` brings its own CUDA and cuDNN libraries, so no CUDA toolkit or library-path setup is needed.
```
conda create -n easymode python=3.10 git -c conda-forge
conda activate easymode

pip install "tensorflow[and-cuda]==2.15.1"
pip install easymode
```

### Older setup: TensorFlow 2.11
easymode also runs on TensorFlow 2.11, the version it was developed with. In that case the CUDA libraries come from conda:
```
# Create environment with CUDA support
conda create -n easymode python=3.10 cudatoolkit=11.2 cudnn=8.1 git -c conda-forge
conda activate easymode

# Install packages
pip install tensorflow==2.11.0 protobuf==3.19.0
pip install easymode

# Set up CUDA library paths (one-time setup)
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d

cat > $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh << 'EOF'
export OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/cuda_env.sh << 'EOF'
export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
unset OLD_LD_LIBRARY_PATH
EOF

# Re-activate to apply CUDA paths
conda deactivate
conda activate easymode

```

To use `--xla` (in `easymode segment` and `easymode train`) in this environment, also add the CUDA compiler tools once:
```
conda install -c nvidia cuda-nvcc=11.3.58
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
ln -sf $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/libdevice.10.bc
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib
```


## Environment settings

### AreTomo3
In case you want to use `easymode reconstruct`, you need to set the path to the AreTomo3 binary and define the required AreTomo3 environment initialization command. For example:

```
easymode set --aretomo3-path /public/EM/AreTomo/AreTomo3 --aretomo3-env "module load AreTomo/3.1.0"
```

### Cache directory
When using `easymode segment <feature>` for the first time, the required model weights will be downloaded and saved to a local cache directory (~500 MB per model). The default location is `~/easymode/`, but you can change it (for example if you want a central cache for all your users) as follows:

```
easymode set --cache-directory /public/easymode/cache/ 
```