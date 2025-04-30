# SOFT
Reducing Calibration Efforts of SSVEP-BCIs by Shallow Fine-Tuning-Based Transfer Learning

# Description
1. Here are the codes of the SOFT in the paper ["Reducing Calibration Efforts of SSVEP-BCIs by Shallow Fine-Tuning-Based Transfer Learning“]().
2. The core code for SOFT can be found in the `finetuning.py` file.

## The related version information
1. Python == 3.9.13
2. Keras-gpu == 2.6.0
3. tensorflow-gpu == 2.6.0
4. scipy == 1.9.3
5. numpy == 1.19.3
## Training for the benchmark dataset
1. Download the code.
2. Download the [benchmark dataset](http://bci.med.tsinghua.edu.cn/download.html) and its [paper](https://ieeexplore.ieee.org/abstract/document/7740878).
3. Create a model folder to save the model.
4. Change the data and model folder paths in train and test files to your data and model folder paths.
