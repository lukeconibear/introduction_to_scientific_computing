# Python - Installation

## Miniconda
Miniconda is a minimal installer for Conda.  
Conda is a package, dependency, and environment management for any language.  

1. Download the latest miniconda:
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ```
    
2. Run bash script, read terms, and set path:
    ```bash
    . Miniconda3-latest-Linux-x86_64.sh
    ```
    
3. Create a python environment with some useful libraries:
    ```bash
    conda create -n python3_teaching -c conda-forge -c oggm numpy xarray cartopy matplotlib jupyterlab pandas geopandas salem scipy dask xesmf scipy geoviews hvplot rasterio affine wrf-python
    ```
    
4. To activate/deactivate conda environment:
    ```bash
    conda activate python3_teaching
    conda deactivate
    ```
   
## Misc.
To list your environments:
```bash
conds info --envs
```

To list the libraries in your activated environment:
```bash
conda list
conda list | grep <package-name>
```

To create an environment from a YAML file:
```bash
conda env create -f my_environment.yml
```

To install a library (you can search for it on Anaconda [here](https://anaconda.org/)):
```bash
conda install <package-name>
conda install -c conda-forge <package-name>
```

Can also use PIP to install packages:
```bash
pip install <package-name>
```

For more information, see the [documentation](https://docs.conda.io/en/latest/miniconda.html).  