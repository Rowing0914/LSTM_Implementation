## Introduction
I have implemented LSTM of 1997 version in python, which incorporates only CEC(Carousel Error Constant).  
As for the detail of the algorithm of LSTM, please refer to the like below.  
This is my summary on LSTM.  
URL: https://qiita.com/Rowing0914/items/1edf16ead2190308ce90  

## Python Version
2.7  

## Training Dataset
data/data.csv : Aggregated Reddit comments and small version of reddit-comments-2015-08.csv  

## Usage
1. Install all requirements below  
pip install --upgrade numpy csv itertools operator nltk sys datetime matplotlib  
2. Run python script: LSTM_1997_CEC.py  

## Comments
Since my objective was for personal study, I did hardcode many parameters in that script, for example word_dim, hidden_dim and so on.  
So If you would like to play with it, feel free to modify it.  