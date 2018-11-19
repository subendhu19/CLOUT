from __future__ import division
import argparse
import math
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pickle

from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm

class RNN(nn.Module):
	def __init__(self, epochs=5, batchsize=50, vocabsize=5, embsize=5):
		super(RNN, self).__init__()
		self.epochs = 5
		self.batchsize = batchsize
		self.vocabsize = vocabsize
		self.embsize = embsize

		self.emb = nn.Linear(vocabsize, embsize)
		self.rnn = nn.LSTM(input_size=embsize, hidden_size=embsize, num_layers=1, dropout=0.05)
		self.out = nn.Linear(embsize, 1)
		self.sig = nn.Sigmoid()

	def forward(self, inputs, hidden=None, force=True, steps=0):
		if force or steps == 0: steps = len(inputs)
		outputs = Variable(torch.zeros(steps, 1, 1))
		inputs = F.relu(self.emb(inputs))
		inputs = inputs.view(inputs.size()[0],1,inputs.size()[1])
		outputs, hidden = self.rnn(inputs, hidden)
		outputs = self.out(outputs)
		return outputs.squeeze(), hidden

	def predict(self, inputs):
		out, hid = self.forward(inputs, None)
		return self.sig(out[-1]).data[0]

n_epochs = 5
vocabsize = 942
embsize = 50

losses = np.zeros(n_epochs) # For plotting

input_seqs = np.array(pickle.load(open('../full data/MIMICIIIPROCESSED.3digitICD9.seqs')))
labels = np.array(pickle.load(open('../full data/MIMICIIIPROCESSED.morts')))

trainratio = 0.7
validratio = 0.1
testratio = 0.2

trainlindex = int(len(input_seqs)*trainratio)
validlindex = int(len(input_seqs)*(trainratio + validratio))
batchsize = 50

# def sample_input():
# 	return [[[1,0],[0,1],[1,1]],[[1,0],[1,0],[0,0],[1,0],[1,0]],[[1,1],[0,1],[0,0],[0,1]],[[1,1],[0,1]],[[0,1],[0,1],[0,1],[1,1],[1,0]]]

# def sample_output():
# 	return [[1],[0],[1],[0],[1]]

def convert_to_one_hot(code_seqs, len=vocabsize):
	new_code_seqs = []
	for code_seq in code_seqs:
		one_hot_vec = np.zeros(len)
		for code in code_seq:
			one_hot_vec[code] = 1
		new_code_seqs.append(one_hot_vec)
	return np.array(new_code_seqs)

best_aucrocs = []
for run in range(10):
    print 'Run', run

    perm = np.random.permutation(input_seqs.shape[0])
    rinput_seqs = input_seqs#[perm]
    rlabels = labels#[perm]

    train_input_seqs = rinput_seqs[:trainlindex]
    train_labels = rlabels[:trainlindex]
    train_labels = train_labels.reshape(train_labels.shape[0],1)

    valid_input_seqs = rinput_seqs[trainlindex:validlindex]
    valid_labels = rlabels[trainlindex:validlindex]

    test_input_seqs = rinput_seqs[validlindex:]
    test_labels = rlabels[validlindex:]

    n_iters = train_input_seqs.shape[0]

    model = RNN(n_epochs, 1, vocabsize, embsize)
    criterion = nn.BCEWithLogitsLoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    aucrocs = []

    for epoch in range(n_epochs):

        print 'Epoch', (epoch+1)

        for i in (range(0, n_iters, batchsize)):
            batch_icd = train_input_seqs[i:i+batchsize]
            batch_train_labels = train_labels[i:i+batchsize]

            optimizer.zero_grad()
            losses = []

            for iter in range(len(batch_icd)):
                icd_onehot = convert_to_one_hot(batch_icd[iter], vocabsize)

                icd_inputs = Variable(torch.from_numpy(icd_onehot).float())

                targets = Variable(torch.from_numpy(batch_train_labels[iter]).float())

                # Use teacher forcing 50% of the time
                force = random.random() < 0.5
                outputs, hidden = model(icd_inputs, None, force)

                #print outputs[-1], targets
                losses.append(criterion(outputs[-1], targets))

            loss = sum(losses)/len(batch_icd)
            loss.backward()
            optimizer.step()

        ## Validation phase
        vpredictions = np.zeros(len(valid_input_seqs))
        for i in range(len(valid_input_seqs)):
            test_seq = valid_input_seqs[i]
            vpredictions[i] = model.predict(Variable(torch.from_numpy(convert_to_one_hot(test_seq)).float()))
        print "Validation Test AUC_ROC: ", roc_auc_score(valid_labels, vpredictions)

        ## Testing phase
        predictions = np.zeros(len(test_input_seqs))
        for i in range(len(test_input_seqs)):
        	test_seq = test_input_seqs[i]
        	predictions[i] = model.predict(Variable(torch.from_numpy(convert_to_one_hot(test_seq)).float()))
        print "Test AUC_ROC: ", roc_auc_score(test_labels, predictions)
        # actual_predictions = (predictions>0.5)*1
        # print classification_report(test_labels, actual_predictions)

        aucrocs.append(roc_auc_score(test_labels, predictions))

    best_aucrocs.append(max(aucrocs))

print "Average AUCROC:", np.mean(best_aucrocs), "+/-", np.std(best_aucrocs)    

    # Use some plotting library
    # if epoch % 10 == 0:
        # show_plot('inputs', _inputs, True)
        # show_plot('outputs', outputs.data.view(-1), True)
        # show_plot('losses', losses[:epoch] / n_iters)

        # Generate a test
        # outputs, hidden = model(inputs, False, 50)
        # show_plot('generated', outputs.data.view(-1), True)

