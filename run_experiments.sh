#!/bin/bash

echo Running the latent space models
python CAE.py
python AE.py
printf "\n"


echo Logistic Regression
python logistic_regression.py
printf "\n"

echo LSTM with only ICD
python rnn_icd.py
printf "\n"

echo LSTM - simple concatenation
python rnn_concat.py
printf "\n"

echo LSTM - only latent CAE
python rnn_latent.py --emb_weights '../full data/CAE/CAE_embedding_weights.npy'
printf "\n"

echo LSTM - only latent AE
python rnn_latent.py --emb_weights '../full data/CAE/AE_embedding_weights.npy'
printf "\n"

echo LSTM - concat latent AE
python rnn_concat_latent.py --emb_weights '../full data/CAE/AE_embedding_weights.npy'
printf "\n"

echo LSTM - concat latent CAE
python rnn_concat_latent.py --emb_weights '../full data/CAE/CAE_embedding_weights.npy'
printf "\n"