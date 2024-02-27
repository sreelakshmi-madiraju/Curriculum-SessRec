# Curriculum-SessRec
## Improved Session-based Recommender Systems using Curriculum Learning

This is the code for the paper "Improved Session-based Recommender Systems using Curriculum Learning" (under review). 

## Datasets

-> The following datasets are used in our experiments. 

YOOCHOOSE: http://2015.recsyschallenge.com/challenge.html or https://www.kaggle.com/chadgostopp/recsys-challenge-2015
Download yoochoose-clicks.dat from the above link and save the file in Datasets folder 

DIGINETICA: http://cikm2016.cs.iupui.edu/cikm-cup or https://competitions.codalab.org/competitions/11161
Download train-item-views.csv from the above link and save the file in Datasets folder

AMAZON: https://jmcauley.ucsd.edu/data/amazon/
Download beauty review data (reviews_Beauty_5.json) from the above link and run pre_process_amazon.py to generate train and test files for AMAZON data

-> Run preprocess.py with dataset name as argument to generate train and test files for YOOCHOOSE and DIGINETICA. (dataset name: diginetica/yoochoose/sample)
Example: python preprocess.py --dataset=yoochoose

-> The required files are already available in the appropriate folders. 

## Compute TransE Embedding 

-> Run create_transe_emb.py to generate KG triples and compute transe embedding. Parameters are optimized based on validation data.

-> The required files are already available in the appropriate folders. 

## Run the models

-> train_with_CL.ipynb has code for training models (LSTM, Transformer) with applied curriculum (C1 or C2 or Hybrid or Adaptive) 

-> train_without_CL.ipynb has code for training models (LSTM, Transformer) without curriculum.

## Requirements

-> TransE embedding is implemented using Ampligraph (https://docs.ampligraph.org/en/1.4.0/generated/ampligraph.latent_features.TransE.html)

-> ampligraph 1.4.0

-> tensorflow or tensorflow-gpu '>=1.15.2,<2.0.0'

-> Python 3

-> keras 2.15.0

