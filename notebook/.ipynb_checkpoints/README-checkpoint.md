### __** This folder contains all the notebooks generated during the project **__

> Since our experiments are performed on Google Colab for higher RAM space and better GPUs availability, most of our development codes are stored in notebooks. 
> * NOTE: these notebooks were downloaded directly from Google Colab, so in order to run most of the notebooks locally, modifications to some of the data paths are needed.

* `stage-1` folder: contains the notebooks running our initial of experiments. More specifically, there are three notebooks
>    * `cnn_1st_attempt.ipynb`: first CNN-based model with MSE and MAE. Only 1 input stream.
>    * `cnn_2nd_attempt.ipynb`: first CNN-based model with MAE. 2 input streams.
>    * `typenet_1st_attempt.ipynb`: first TypeNet-based model with MAE. 2 input streams.


* `stage-2` folder: contains the notebooks running our second set of experiments. In this phase, we focused on streamline the preprocessing stage and understanding the "Average User" profile. We also tried out different model structures and losses.

* `data_exploration.ipynb`: a notebook to gain a quick feel of our original dataset, meta dataset, extracted features, and input data.

* `KDR_Preprocessing.ipynb`: the notebook where we developed our preprocessing codes (the newest versions are stored now in the `helper.py` file). In particular, historical versions of the preprocessing functions are stored in this notebook as comments.

* `Model_Architecture_List.ipynb`: the notebook where we keep track of the model structures. Almost all changes to the model, including modifying regularizer or dropout rates, are generally saved into a new version.

* `stage_3_experiments.ipynb`: the main notebook containing training print-outs, results of our most recent set of experiments. Some of the experiments are no longer in the notebook, but the details and configurations of the experiments are all stored [here](results/experiments_tracking_details/stage_3_exp_details.html).

* `tensorboard-demo.ipynb`: contains the relative path to all tensorboard logs for visualization. This notebook can be ran locally to view our tensorboard results. NOTE: the models generally converges rather quickly, this could be because our training data sizes are relatively large.