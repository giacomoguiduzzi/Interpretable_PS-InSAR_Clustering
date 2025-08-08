# Interpretable Clustering of PS-InSAR Time Series for Ground Deformation Detection

This is the repo for the
paper "[Interpretable Clustering of PS-InSAR Time Series for Ground Deformation Detection](https://link.tbd)"
by Claudia Masciulli, Giacomo Guiduzzi, Donato Tiano, Marta Zocchi,
Francesco Guerra, Paolo Mazzanti and Gabriele Scarascia Mugnozza.

The repo is structured as follows:

- `data`: contains the data used in the paper
- `src`: contains the source code
- `time2feat`: contains the code for the [Time2Feat](https://github.com/softlab-unimore/time2feat) model
- `featts`: contains the code for the [FeaTTS](https://github.com/protti/FeatTS) model
- `jqm_cvi`: contains the code for the [JQM-CVI](https://github.com/jqmviegas/jqm_cvi) package, which implements the
  Dunn index for clustering evaluation among other metrics.

Both models are available as Python packages on PyPI and are configured as git submodules in this repo.

The suggested way to prepare the code is:

1. Run `git submodule init --recursive` to initialize time2feat, FeatTS and JQM-CVI submodules.
    - Alternatively, you can clone the repo with `git clone --recurse-submodules`
2. Install the packages contained in the requirements.txt file into a Python venv or conda environment
3. Move into the `jqm_cvi` folder and run `python setup.py install` to install the JQM-CVI package for the Dunn index
4. Move back into the repo folder and export `PYTHONPATH=/path/to/Interpretable_PS-InSAR_Clustering` to let Python see
   the `time2feat` and `FeatTS` folders as Python modules; finally run `python src/simple_pipeline.py` from inside the
   repo folder.
    - Alternatively, directly run `PYTHONPATH=/path/to/Interpretable_PS-InSAR_Clustering python src/simple_pipeline.py`
      from inside the repo folder.

For any need related to the code, feel free to open a GitHub issue or email me at: giacomo.guiduzzi at unimore.it.
