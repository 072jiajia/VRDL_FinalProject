# VRDL Final Project
Code for Private Score 0.9270 in Bengali.AI Handwritten Grapheme Classification

## Abstract
In this work, I use API-Net to train my model<br>
API-Net [Paper](https://arxiv.org/pdf/2002.10191.pdf) | [GitHub](https://github.com/PeiqinZhuang/API-Net)

## Hardware
The following specs were used to create the solutions.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
- 3x GeForce RTX 2080 Ti

## Reproducing Submission
To reproduct my submission, run the code my [Notebook in Kaggle](https://www.kaggle.com/jia072/bengali-ai)

## Producing Your Own Submission
To produce your own submission, do the following steps:
1. [Installation](#installation)
2. [Prepare Data](#dataset-preparation)
3. [Train and Make Submission](#train-and-make-prediction)


## Installation
All requirements should be detailed in requirements.txt. Using virtual environment is recommended.
```
virtualenv .
source bin/activate
pip3 install -r requirements.txt
```

## Dataset Preparation
You need to download the [training data](https://www.kaggle.com/c/bengaliai-cv19/data) by yourself.<br>
Unzip the zip file and directory is structured as:
```
VRDL_FinalProject
  +- train.csv
  +- train_image_data_0.parquet
  +- train_image_data_1.parquet
  +- train_image_data_2.parquet
  +- train_image_data_3.parquet
  +- test.csv
  +- test_image_data_0.parquet
  +- test_image_data_1.parquet
  +- test_image_data_2.parquet
  +- test_image_data_3.parquet
  .
  .
  .
```


## Train and Make Prediction
You can simply run the following command to train your models.
```
$ python train_root.py
$ python train_vowel.py
$ python train_consonant.py
```

The expected training time is:

GPUs | Training Epochs | Training Time
------------- | ------------- | -------------
3x 2080 Ti | 50 | 24 hours

After we finish training our models, upload your file to kaggle and make submission.
Here is my [Notebook in Kaggle](https://www.kaggle.com/jia072/bengali-ai)

## Citation
```
@inproceedings{zhuang2020learning,
  title={Learning Attentive Pairwise Interaction for Fine-Grained Classification.},
  author={Zhuang, Peiqin and Wang, Yali and Qiao, Yu},
  booktitle={AAAI},
  pages={13130--13137},
  year={2020}
}
```
