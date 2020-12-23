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
To reproduct my submission, you can see my [Notebook in Kaggle](https://www.kaggle.com/jia072/bengali-ai)

## Producing Your Own Submission
To produce your own submission, do the following steps:
1. [Prepare Data](#dataset-preparation)
2. [Installation](#installation)
3. [Train and Make Submission](#train-and-make-prediction)


## Dataset Preparation
You need to download the [training data](https://www.kaggle.com/c/bengaliai-cv19/data) by yourself.<br>
And put the zip file into the same directory as main.py, the directory is structured as:
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

## Installation
All requirements should be detailed in requirements.txt. Using virtual environment is recommended.
```
virtualenv .
source bin/activate
pip3 install -r requirements.txt
```



## Train and Make Prediction
You can simply run the following command to train your models and make submission.
```
$ python main.py
```
If you'd like to train in custom hyperparameters, change the hyperparameters to whatever you like.<br>
Or you may try the following command.
```
$ python main.py --exp_name=custom --epochs==50 --KFold=3 --n_classes=20 --n_samples=3
```

You may interrupt your program at any time.<br>
(ex: You're sharing the GPUs with your classmates and they think that you're using too many GPUs or you have occupied them for too long.)

This code records the checkpoint in every epoch, so you can just input the same command line to resume the code.<br>
The expected training time is:

GPUs | KFold | Image size | Training Epochs | Training Time
------------- | ------------- | ------------- | ------------- | -------------
4x 2080 Ti | 10 | 320 | 100 | 19 hours

In main.py, after we finish training K models, it will directly call
```
python3 get_answer.py {your exp_name}
```
It will generate a file {your exp_name}.csv which is the prediction of the testing dataset<br>
Use the csv file to make your submission!

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
