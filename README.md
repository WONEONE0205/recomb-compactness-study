# recomb-compactness-study
This repository contains the Python code used for the dissertation *The Effects of Compactness Weighting on ReComb-Generated Redistricting Ensembles*.

## Files
src/load data.py: loads and inspects the Pennsylvania VTD dataset.
src/toy model.py: implements the simplified ReComb illustration.
src/pa_recom_experiment.py: runs the Pennsylvania ReComb experiments.
src/trajectory.py: processes the experimental results and generates figures.
data/PA_VTDs.json: contains the Pennsylvania VTD data used in the study.

## Experimental settings

The experiments used four compactness weights, beta = 0, 0.05, 0.1, and 0.5, and three random seeds: 42, 123, and 1234.Each chain contained 10,000 transitions in addition to the initial state. Checkpoints were saved every 1,000 transitions.

To run a particular chain, set 'BETA' and 'RANDOM_SEED in 'pa_recom_experiment.py' to the required values.

Before running 'toy model.py , replace 'FIGURES_DIR' with the required output location. Before running 'trajectory.py', replace 'RESULTS_ROOT' with the folder containing the saved experimental results.

The required Python packages are listed in 'requirements.txt'.
