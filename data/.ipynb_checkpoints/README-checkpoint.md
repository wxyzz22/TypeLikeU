### __** This folder contains the dataset that is used in our project **__

> * The original dataset is downloadable from this [website](https://userinterfaces.aalto.fi/136Mkeystrokes/). 
> * The file is a _.zip_ file and is 1.4GB zipped, __16GB__ unzipped.

The dataset was collected using an online typing test following scientific standards for testing typing performance as a result of the paper [Observations on Typing from 136 Million Keystrokes](https://userinterfaces.aalto.fi/136Mkeystrokes/resources/chi-18-analysis.pdf) by Vivek Dhakal, Anna Maria Feit, Per Ola Kristensson, and Antti Oulasvirta published in 2017. Each participant's keystroke data is stored in a _.txt_ file of the form `PARTICIPANT_ID_keystrokes.txt`, where `PARTICIPANT_ID` is a unique id assigned to each participant in the data collection. For more information on the data, refer to [this](TypeLikeU_COMP576_FinalProject/data/meta_data/readme.txt).

For the purpose of shortening training time and limited training resources, we are only using all the users with `PARTICIPANT_ID` $\leq$ `7001` in the project, and this partial dataset is provided in the folder `keystroke_data_7001`. 

The `.txt` files of users keystroke data have already been processed; in particular, the following three files contains mis-aligned rows and have been replaced with correct formatted ones:
    * `562_keystrokes.txt`: `KEYSTROKE_ID` = 270318
    * `3106_keystrokes.txt`: `KEYSTROKE_ID` = 1560722
    * `4200_keystrokes.txt`: `KEYSTROKE_ID` = 2131138

If you download the original _.zip_ file from the [website](https://userinterfaces.aalto.fi/136Mkeystrokes/), pay attention to the buggy files when running our `processing_folder` function in the [helper.py](TypeLikeU_COMP576_FinalProject/helper.py) file.