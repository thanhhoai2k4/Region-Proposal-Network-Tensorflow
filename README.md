# Region-Proposal-Network-Tensorflow
The repository is used to implement a Region Proposal Network (RPN) in Faster R-CNN

## 1. Requirement
* Python :[3.10](https://www.python.org/downloads/release/python-31016/)
* Tensorflow 2.17.1: needed CUDA.
* IDE: Pycharm.


## 2. Data Preparation

I get data from Kaggle. This is a dataset of approximately 800 images of wearing masks, not wearing masks, and wearing masks improperly.
You can get data From [Here](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection). Pleased see [example](https://github.com/thanhhoai2k4/Region-Proposal-Network-Tensorflow/tree/main/data_training).

## 3. create environment

in Current Path, you install package:

    `pip install -r requirements.txt`

## 4.Training with tensorflow.

"Training_Cache.py" : this is training file with a small dataset.

"Trainning_Generator.py" : this is training file using [tf.Data](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) is parallel load.

"Trainning_pre.py" : this is pre-Training.




## 5. result.

![Alt text](result_01.png)