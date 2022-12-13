# TypeLikeU

Team member(s): Xingya Wang

This is the repository containing code and results of our final project for the Fall 2022, [COMP576: Introduction to Deep Learning](http://elec576.rice.edu/) course at Rice University. This project aims to tackle the regression problem associated to users' keystroke dynamics: more concretely, we want to predict the users' keytroke behavior on new keys given their historical typing data.


### Navigating the Repo
The details of our final project are organized into four folders and a `helper.py` file in this repository. Here are some brief descriptions of each to help with navigation:

* The `data` folder: contains sample datasets that we used in the project. They are all sub-samples from the open-source dataset provided [here](https://userinterfaces.aalto.fi/136Mkeystrokes/). For a more detailed description of files in the folder, refer to [this README](data/README.md).

* The `logs` folder: contains model training logs, including tensorboard records, checkpoint file, and train-val loss plots. For a more detailed description of files in the folder, refer to [this README](logs/README.md).

* The `notebook` folder: since our experiments are performed over the Google Colab platform for higher RAM space and better GPUs availability, most of our development codes for preprocessing, model structures, model trainings and etc. are all stored in notebooks. In particular, we also included a [_data_exploration.ipynb_](notebook/data_exploration.ipynb) notebook to provide a quick feel of our dataset. For a more detailed description of files in the folder, refer to [this README](notebook/README.md).

* The `results` folder: contains our [final report](results/report.ipynb), [final poster](results/poster.png), and [experiment results](results/experiments_tracking_details) which are stored in _.html_ files for quick summary overview. For a more detailed description of files in the folder, refer to [this README](results/README.md).

* The `helper.py` file: this is the main module we used to run experiments. It contains functions and objects to perform 
    * importing the datasets
    * encode keyboard layout
    * extract features (e.g. uni/di-graphs, time latencies, keycode-pair distances, average user profile)
    * forming sequential inputs (KDS object)
    * forming image-like inputs (KDI object)
    * functionalized callbacks for assisting model training: early stopping, checkpoints, learning rate scheduler, tensorboard records.
    * miscellaneous functions for visualization

### Results Overview

By the end of this final project, we achieved the following:
* the preprocessing steps, as well as results recording steps, are completely streamlined and stored in `helper.py` file
* we considered two major input formats to the model, and proposed a __new__ feature engineering method -- the __two-channel-KDI input format__ (see [__Section 5.4__](#54)), which works substantially better than traditional sequential inputs
* the structure of the general modeling pipeline are largely determined: two channels of inputs are needed, either sequential or image-like input formats, where one encodes the "historical user data" and is fed into the "User Embedding Layer", and the other encodes the "current keycode" and is fed into the "Keycode Embedding Layer". The outputs from both layers are "concatenated" (in some way) and fed into a common "Concat Layer" for generating time latencies predictions:
<center>
    <img src="results/img/two-channel-KDI-inputs.png" alt="two-channel-KDI input format" width="850"/>
</center>
* we constructed a couple model structures (for both of the embedding layers, as well as the concat output layer, refer to details [here](notebook/Model_Architecture_List.ipynb)) that perform relatively well: 
    * training loss (MAE) = 57.72 (in milliseconds)
    * validation loss (MAE) = 47.04 (in milliseconds)
    * testing loss (MAE) = 46.8 (in milliseconds)
* we proposed concrete next-steps for future exploration
