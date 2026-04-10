# Installation

## Installing from scratch
The following command will create a new environment called `easymode` and install easymode and all dependencies into it. 
```
# Create environment with CUDA support
conda create -n easymode python=3.10 cudatoolkit=11.2 cudnn=8.1 git -c conda-forge
conda activate easymode

# Install packages
pip install tensorflow==2.11.0 protobuf==3.20.3
pip install git+https://github.com/bionanopatterning/Ais.git
pip install git+https://github.com/mgflast/easymode.git
pip install git+https://github.com/bionanopatterning/Pom.git

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