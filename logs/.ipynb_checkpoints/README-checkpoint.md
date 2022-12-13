### __** This folder contains the training logs of experiments ran in the notebooks **__

> Since our experiments are performed on Google Colab for higher RAM space and better GPUs availability, most of our development codes are stored in notebooks. This folder contains the training results and model checkpoints for the corresponding notebooks.
> * For a complete list of correspondence between notebooks and training logs, refer to [this file](../results/experiments_tracking_details/older_exp_details.html)

* both the `training-logs` folder and the `experiments` folder contains two subfolders (below), and inside each subfolder, there are sub-subfolders with model names containing the respective files
    * `checkpoints`
    * `tensorboard`: run to [this notebook](../notebook/tensorboard-demo.ipynb) for a quick overview of the tensorboard results

* the `loss-plots` folder are the loss plots, and some learning rate versus loss plots, of the corresponding experiment id in the picture name.
    * These are experiments carried out in Stage-3, for codes and training progress printout, refer to the [original notebook](../notebook/stage_3_experiments.ipynb). Note: Stage-3 contains the most recent groups of experiments.