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
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, classification_report

class RNN(nn.Module):
	def __init__(self, epochs=5, batchsize=50, vocabsize=5, embsize=100):
		super(RNN, self).__init__()
		self.epochs = 5
		self.batchsize = batchsize
		self.vocabsize = vocabsize
		self.embsize = embsize

		self.emb_icd = nn.Linear(vocabsize_icd, embsize_icd)
		self.emb_meds = nn.Linear(vocabsize_meds, embsize_meds)
		self.emb_labs = nn.Linear(vocabsize_labs, embsize_labs)

		self.rnn = nn.LSTM(input_size=embsize, hidden_size=embsize, num_layers=1, dropout=0.05)
		self.out = nn.Linear(embsize, 1)
		self.sig = nn.Sigmoid()

	def forward(self, input_icd, input_med, input_lab, hidden=None, force=True, steps=0):
		if force or steps == 0: steps = len(input_icd)
		outputs = Variable(torch.zeros(steps, 1, 1))

		input_icd = F.relu(self.emb_icd(input_icd))
		input_med = F.relu(self.emb_meds(input_med))
		input_lab = F.relu(self.emb_labs(input_lab))

		inputs = torch.cat((input_icd, input_med, input_lab),1)

		inputs = inputs.view(inputs.size()[0],1,inputs.size()[1])
		outputs, hidden = self.rnn(inputs, hidden)
		outputs = self.out(outputs)
		return outputs.squeeze(), hidden

	def predict(self, input_icd, input_med, input_lab):
		out, hid = self.forward(input_icd, input_med, input_lab, None)
		return self.sig(out[-1]).data[0]

n_epochs = 5
vocabsize_icd = 942
vocabsize_meds = 3202
vocabsize_labs = 284 #all 681
vocabsize = vocabsize_icd+vocabsize_meds+vocabsize_labs

embsize_icd = 50
embsize_meds = 75
embsize_labs = 50
embsize = embsize_icd + embsize_labs + embsize_meds

input_seqs_icd = np.array(pickle.load(open('../full data/MIMICIIIPROCESSED.3digitICD9.seqs')))
input_seqs_meds = np.array(pickle.load(open('../full data/MIMICIIIPROCESSED.meds.seqs')))
input_seqs_labs = np.array(pickle.load(open('../full data/MIMICIIIPROCESSED.abnlabs.seqs')))
input_seqs_fullicd = np.array(pickle.load(open('../full data/MIMICIIIPROCESSED.seqs')))

icditems = pickle.load(open('../full data/type dictionaries/MIMICIIIPROCESSED.types', 'rb'))
meditems = pickle.load(open('../full data/type dictionaries/MIMICIIIPROCESSED.meds.types', 'rb'))
labitems = pickle.load(open('../full data/type dictionaries/MIMICIIIPROCESSED.abnlabs.types', 'rb'))

interpretation_file = open("RNN_Concat_Interpretations.txt", 'w')

# overall_risk_factor_file = open("risk_factors_averaged.txt", "w")

labnames = {}
lab_dict_file = open('D_LABITEMS.csv', 'r')
lab_dict_file.readline()
for line in lab_dict_file:
	tokens = line.strip().split(',')
	labnames[tokens[1].replace('"','')] = tokens[2]
lab_dict_file.close()

icdnames = {}
icd_dict_file = open('D_ICD_DIAGNOSES.csv', 'r')
icd_dict_file.readline()
for line in icd_dict_file:
	tokens = line.strip().split(',')
	icdnames[tokens[1].replace('"','')] = tokens[2]
icd_dict_file.close()

icd_scores = {}
med_scores = {}
lab_scores = {}

icd_totals = {}
med_totals = {}
lab_totals = {}


def get_ICD(icd):
	ret_str = ""
	icd_str = icditems.keys()[icditems.values().index(icd)]
	actual_key = icd_str.replace(".", "")[2:]
	if actual_key in icdnames:
		ret_str = icdnames[actual_key]
	else:
		ret_str = icd_str
	return ret_str

def get_med(med):
	ret_str = meditems.keys()[meditems.values().index(med)]
	return ret_str

def get_lab(lab):
	ret_str = labnames[labitems.keys()[labitems.values().index(lab)]]
	return ret_str


print 'Data loaded..'

labels = np.array(pickle.load(open('../full data/MIMICIIIPROCESSED.morts')))

trainratio = 0.7
validratio = 0.1
testratio = 0.2

trainlindex = int(len(input_seqs_icd)*trainratio)
validlindex = int(len(input_seqs_icd)*(trainratio + validratio))

print 'Data prepared..'

def convert_to_one_hot(code_seqs, len):
	new_code_seqs = []
	for code_seq in code_seqs:
		one_hot_vec = np.zeros(len)
		for code in code_seq:
			one_hot_vec[code] = 1
		new_code_seqs.append(one_hot_vec)
	return np.array(new_code_seqs)

def get_factors(icd_seq, med_seq, lab_seq, model, actual_score, full_icd):
	potential_test_data = []

	for seq in range(len(icd_seq)):
		for i in range(len(icd_seq[seq])):
			potential_test_data.append(("icd", full_icd[seq][i], seq, icd_seq[:seq]+[icd_seq[seq][:i] + icd_seq[seq][i+1:]]+icd_seq[seq+1:], med_seq, lab_seq))
	for seq in range(len(med_seq)):
		for i in range(len(med_seq[seq])):
			potential_test_data.append(("med", med_seq[seq][i], seq, icd_seq, med_seq[:seq]+[med_seq[seq][:i]+med_seq[seq][i+1:]]+med_seq[seq+1:], lab_seq))
	for seq in range(len(lab_seq)):
		for i in range(len(lab_seq[seq])):
			potential_test_data.append(("lab", lab_seq[seq][i], seq, icd_seq, med_seq, lab_seq[:seq]+[lab_seq[seq][:i] + lab_seq[seq][i+1:]]+lab_seq[seq+1:]))

	risk_scores = []

	for pt in potential_test_data:
		test_input_icd = Variable(torch.from_numpy(convert_to_one_hot(pt[3], vocabsize_icd)).float())
		test_input_med = Variable(torch.from_numpy(convert_to_one_hot(pt[4], vocabsize_meds)).float())
		test_input_lab = Variable(torch.from_numpy(convert_to_one_hot(pt[5], vocabsize_labs)).float())
		factor_score = actual_score - model.predict(test_input_icd, test_input_med, test_input_lab)
		factor = ""
		if pt[0] == 'icd':
			icd_tag = get_ICD(pt[1])
			factor = "ICD-"+icd_tag
			if icd_tag in icd_scores:
				icd_scores[icd_tag] += factor_score
				icd_totals[icd_tag] += 1
			else:
				icd_scores[icd_tag] = factor_score
				icd_totals[icd_tag] = 1
		elif pt[0] == 'med':
			med_tag = get_med(pt[1])
			factor = "Med-"+med_tag
			if med_tag in med_scores:
				med_scores[med_tag] += factor_score
				med_totals[med_tag] += 1
			else:
				med_scores[med_tag] = factor_score
				med_totals[med_tag] = 1
		else:
			lab_tag = get_lab(pt[1])
			factor = "Lab-"+lab_tag
			if lab_tag in lab_scores:
				lab_scores[lab_tag] += factor_score
				lab_totals[lab_tag] += 1
			else:
				lab_scores[lab_tag] = factor_score
				lab_totals[lab_tag] = 1
		risk_scores.append(("Encounter-"+str(pt[2])+": "+factor, factor_score))

	risk_scores.sort(key=lambda tup: tup[1], reverse=True)

	return risk_scores[:10]

print 'Starting training..'

batchsize = 50

best_aucrocs = []
for run in range(10):
	print 'Run', run

	perm = np.random.permutation(input_seqs_icd.shape[0])
	rinput_seqs_icd = input_seqs_icd#[perm]
	rinput_seqs_meds = input_seqs_meds#[perm]
	rinput_seqs_labs = input_seqs_labs#[perm]
	rinput_seqs_fullicd = input_seqs_fullicd#[perm]
	rlabels = labels#[perm]

	train_input_seqs_icd = rinput_seqs_icd[:trainlindex]
	train_input_seqs_meds = rinput_seqs_meds[:trainlindex]
	train_input_seqs_labs = rinput_seqs_labs[:trainlindex]
	train_labels = rlabels[:trainlindex]
	train_labels = train_labels.reshape(train_labels.shape[0],1)

	valid_input_seqs_icd = rinput_seqs_icd[trainlindex:validlindex]
	valid_input_seqs_meds = rinput_seqs_meds[trainlindex:validlindex]
	valid_input_seqs_labs = rinput_seqs_labs[trainlindex:validlindex]
	valid_labels = rlabels[trainlindex:validlindex]

	test_input_seqs_icd = rinput_seqs_icd[validlindex:]
	test_input_seqs_meds = rinput_seqs_meds[validlindex:]
	test_input_seqs_labs = rinput_seqs_labs[validlindex:]
	test_input_seqs_fullicd = rinput_seqs_fullicd[validlindex:]

	test_labels = rlabels[validlindex:]

	n_iters = train_input_seqs_icd.shape[0]

	model = RNN(n_epochs, 1, vocabsize, embsize)
	criterion = nn.BCEWithLogitsLoss(size_average=False)
	optimizer = optim.Adam(model.parameters(), lr=0.01)

	aucrocs = []

	for epoch in range(n_epochs):

		epoch_loss = 0
		
		print 'Epoch', (epoch+1)

		for i in (range(0, n_iters, batchsize)):
			batch_icd = train_input_seqs_icd[i:i+batchsize]
			batch_meds = train_input_seqs_meds[i:i+batchsize]
			batch_labs = train_input_seqs_labs[i:i+batchsize]

			batch_train_labels = train_labels[i:i+batchsize]

			optimizer.zero_grad()
			losses = []

			for iter in range(len(batch_icd)):
				icd_onehot = convert_to_one_hot(batch_icd[iter], vocabsize_icd)
				med_onehot = convert_to_one_hot(batch_meds[iter], vocabsize_meds)
				lab_onehot = convert_to_one_hot(batch_labs[iter], vocabsize_labs)

				icd_inputs = Variable(torch.from_numpy(icd_onehot).float())
				med_inputs = Variable(torch.from_numpy(med_onehot).float())
				lab_inputs = Variable(torch.from_numpy(lab_onehot).float())

				targets = Variable(torch.from_numpy(batch_train_labels[iter]).float())

				# Use teacher forcing 50% of the time
				force = random.random() < 0.5
				outputs, hidden = model(icd_inputs, med_inputs, lab_inputs, None, force)

				#print outputs[-1], targets
				losses.append(criterion(outputs[-1], targets))

			loss = sum(losses)/len(batch_icd)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.data[0]

		#print(epoch, epoch_loss)

		## Validation phase
		vpredictions = np.zeros(len(valid_input_seqs_icd))
		for i in range(len(valid_input_seqs_icd)):
			test_input_icd = Variable(torch.from_numpy(convert_to_one_hot(valid_input_seqs_icd[i], vocabsize_icd)).float())
			test_input_med = Variable(torch.from_numpy(convert_to_one_hot(valid_input_seqs_meds[i], vocabsize_meds)).float())
			test_input_lab = Variable(torch.from_numpy(convert_to_one_hot(valid_input_seqs_labs[i], vocabsize_labs)).float())
			vpredictions[i] = model.predict(test_input_icd, test_input_med, test_input_lab)

		print "Validation AUC_ROC: ", roc_auc_score(valid_labels, vpredictions)

		## Testing phase
		predictions = np.zeros(len(test_input_seqs_icd))

		# ICD_wise_corr = np.zeros(5)
		# meds_wise_corr = np.zeros(5)
		# labs_wise_corr = np.zeros(5)
		# ICD_wise_tot = np.zeros(5)
		# meds_wise_tot = np.zeros(5)
		# labs_wise_tot = np.zeros(5)

		for i in range(len(test_input_seqs_icd)):
			test_input_icd = Variable(torch.from_numpy(convert_to_one_hot(test_input_seqs_icd[i], vocabsize_icd)).float())
			test_input_med = Variable(torch.from_numpy(convert_to_one_hot(test_input_seqs_meds[i], vocabsize_meds)).float())
			test_input_lab = Variable(torch.from_numpy(convert_to_one_hot(test_input_seqs_labs[i], vocabsize_labs)).float())
			predictions[i] = model.predict(test_input_icd, test_input_med, test_input_lab)
			
			# ICD_wise_corr[get_avg(test_input_seqs_icd[i], 'i')] += int((predictions[i]>0.5)*1 == test_labels[i])
			# ICD_wise_tot[get_avg(test_input_seqs_icd[i], 'i')] += 1

			# meds_wise_corr[get_avg(test_input_seqs_meds[i], 'm')] += int((predictions[i]>0.5)*1 == test_labels[i])
			# meds_wise_tot[get_avg(test_input_seqs_meds[i], 'm')] += 1

			# labs_wise_corr[get_avg(test_input_seqs_labs[i], 'l')] += int((predictions[i]>0.5)*1 == test_labels[i])
			# labs_wise_tot[get_avg(test_input_seqs_labs[i], 'l')] += 1

		print "Test AUC_ROC: ", roc_auc_score(test_labels, predictions)
		
		aucrocs.append(roc_auc_score(test_labels, predictions))
		# actual_predictions = (predictions>0.5)*1
		# print classification_report(test_labels, actual_predictions)

	best_aucrocs.append(max(aucrocs))

print "Average AUCROC:", np.mean(best_aucrocs), "+/-", np.std(best_aucrocs) 		

# print "Final testing and interpretations"
predictions = np.zeros(len(test_input_seqs_icd))
for i in (range(len(test_input_seqs_icd))):
	test_input_icd = Variable(torch.from_numpy(convert_to_one_hot(test_input_seqs_icd[i], vocabsize_icd)).float())
	test_input_med = Variable(torch.from_numpy(convert_to_one_hot(test_input_seqs_meds[i], vocabsize_meds)).float())
	test_input_lab = Variable(torch.from_numpy(convert_to_one_hot(test_input_seqs_labs[i], vocabsize_labs)).float())
	
	test_score = model.predict(test_input_icd, test_input_med, test_input_lab)
	predictions[i] = test_score
	top_risk_factors = get_factors(test_input_seqs_icd[i], test_input_seqs_meds[i], test_input_seqs_labs[i], model, test_score, test_input_seqs_fullicd[i]) 
	if (test_score>0.5):
		interpretation_file.write("ID: " + str(i) + " True label: "+str(test_labels[i])+"\n")
		for rf in top_risk_factors:
			interpretation_file.write(str(rf)+"\n")
		interpretation_file.write("\n")

interpretation_file.close()

# overall_risk_factor_file.write('ICD codes:\n')

# icd_averages = {}
# med_averages = {}
# lab_averages = {}

# for key in icd_scores:
# 	icd_averages[key] = icd_scores[key]/icd_totals[key]

# for key in med_scores:
# 	med_averages[key] = med_scores[key]/med_totals[key]

# for key in lab_scores:
# 	lab_averages[key] = lab_scores[key]/lab_totals[key]

# for key, value in sorted(icd_averages.iteritems(), key=lambda (k,v): (v,k), reverse=True):
#     overall_risk_factor_file.write(str(key) + "-" + str(value) + "\n")
# overall_risk_factor_file.write('\n')   

# overall_risk_factor_file.write('Medications:\n')
# for key, value in sorted(med_averages.iteritems(), key=lambda (k,v): (v,k), reverse=True):
#     overall_risk_factor_file.write(str(key) + "-" + str(value) + "\n")
# overall_risk_factor_file.write('\n')  

# overall_risk_factor_file.write('Lab components:\n')
# for key, value in sorted(lab_averages.iteritems(), key=lambda (k,v): (v,k), reverse=True):
#     overall_risk_factor_file.write(str(key) + "-" + str(value) + "\n")
# overall_risk_factor_file.write('\n')  

# pickle.dump(model, open('mortality_model.p', 'wb'))
# test_patients = []

# for i in range(len(test_input_seqs_icd)):
# 	 test_patients.append((test_input_seqs_icd[i], test_input_seqs_meds[i], test_input_seqs_labs[i], test_input_seqs_fullicd[i], test_labels[i]))

# pickle.dump(test_patients, open('test_patients.p', 'wb'))
# pickle.dump(model._modules['emb_icd'].weight.data.numpy().T, open('icd_embedding_weights.npy', 'wb'))

