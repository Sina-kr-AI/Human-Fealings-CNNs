# Human fealing classificaion (dataset:Human Fealings) 

## Overview

- Classification images
- CNNs Model
- Main code is in the ``src.ipynb``
- Structure of dataset built in ``data_structure.ipynb``
- Dataset link: https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions

## Challenge

- In ``src.ipynb``, during training, the model showed signs of overfitting.
This means the model performed well on the training data but its accuracy on validation/test data decreased.
To address this issue, techniques such as data augmentation, dropout, and regularization can be considered in future improvements.

- The overfitting has removed in ``second_src.ipynb`` as a notebook and ``second_src.py`` as code
  - tips:
    - batchnormalization
    - dropout
    - data augmentation
  
- **Plots**

  - Before correction:
    <img src='puictures/bef loss.png'>
    <img src='puictures/bef acc.png'>

  - After correction:
    <img src='puictures/aft loss.png'> 
    <img src='puictures/aft accuracy.png'>

## Dataset structure

- **new_dataset**
  - train data
    - Angry
    - Fear
    - Happy
    - Sad
    - Suprise
  - val data
    - Angry
    - Fear
    - Happy
    - Sad
    - Suprise
  - test data
    - Angry
    - Fear
    - Happy
    - Sad
    - Suprise
