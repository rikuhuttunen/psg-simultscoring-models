# psg-simultscoring-models

Model definitions used in the manuscript by Huttunen et. al titled "A Comparison of Signal Combinations for Deep Learning-Based Simultaneous Sleep Staging and Respiratory Event Detection"

The models are based on https://github.com/perslev/U-Time.

The code to create the Keras models is provided in `psg_simultscoring_models/utime.py`. The code is provided as a python package, which can be installed by cloning this repository and running `pip install .` in the repository root.

An example on how to install the model creation code, and to instantiate each of the models, is provided in `notebooks/setup_model_architectures.ipynb`.

Trained models are not available, since sensitive patient data was used to train the models.

## Architecture

![architecture](https://github.com/rikuhuttunen/psg-simultscoring-models/blob/b53c8636685348f67bd87b0688a23961ed753ebb/img/paper2_arch_v6.png?raw=true)
