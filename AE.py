from __future__ import division
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

vocabsize_icd = 942
vocabsize_meds = 3202
vocabsize_labs = 284 #all 681 284

class AE(nn.Module):
	def __init__(self, epochs=5, batchsize=50, embsize=100):
		super(AE, self).__init__()
		self.epochs = epochs
		self.batchsize = batchsize
		self.embsize = embsize

		self.emb = nn.Linear(vocabsize_icd + vocabsize_meds + vocabsize_labs, self.embsize)

		self.out = nn.Linear(self.embsize, vocabsize_icd + vocabsize_meds + vocabsize_labs)

		self.reconloss = nn.MSELoss(size_average=True)

	def forward(self, input_icd, input_med, input_lab):

		input_full = torch.cat((input_icd, input_med, input_lab),1)

		hidden_full = F.relu(self.emb(input_full))	 

		output_full = F.relu(self.out(hidden_full))

		return [output_full, hidden_full]

	def get_encodings(self, ICD_data, Lab_data):
		return self.forward(Variable(torch.from_numpy(ICD_data).float()), Variable(torch.from_numpy(Lab_data).float()))[-1]

	def fit(self, ICDs, Meds, Labs):

		optimizer = optim.Adam(self.parameters(), 0.01)

		prev_loss = 1000
		for epoch in range(self.epochs):
			print 'Epoch:', epoch

			perm = np.random.permutation(ICDs.shape[0])
			ICDs = ICDs[perm]
			Meds = Meds[perm]
			Labs = Labs[perm]

			losses = []

			for i in range(0, ICDs.shape[0], self.batchsize):
				ICDbatch, Medbatch, Labbatch = ICDs[i:i+self.batchsize], Meds[i:i+self.batchsize], Labs[i:i+self.batchsize]
				ICDbatchvar, Medbatchvar, Labbatchvar = Variable(torch.from_numpy(ICDbatch).float()), Variable(torch.from_numpy(Medbatch).float()), Variable(torch.from_numpy(Labbatch).float())

				outputs = self.forward(ICDbatchvar, Medbatchvar, Labbatchvar)

				loss = self.reconloss(outputs[0], torch.cat((ICDbatchvar, Medbatchvar, Labbatchvar),1))

				losses.append(loss.data[0])

				optimizer.zero_grad()
			    
				loss.backward()
			    
				optimizer.step()
				# print 'recon loss:', loss_recon.data[0], 'loss_cr:', loss_cr.data[0]

			print 'Epoch loss:', np.mean(losses)

			if abs(np.mean(losses) - prev_loss) < 0.00005:
				break

			prev_loss = np.mean(losses)

model = AE(10,50,175)
ICD_data = pickle.load(open('../full data/CAE/CAEEntries.3digitICD9','r'))
Med_data = pickle.load(open('../full data/CAE/CAEEntries.meds','r'))
Lab_data = pickle.load(open('../full data/CAE/CAEEntries.abnlabs','r'))
model.fit(ICD_data, Med_data, Lab_data)

emb_weights = model._modules['emb'].weight.data.numpy().T
print 'Pickled embedding weights. Shape:', np.array(emb_weights).shape
pickle.dump(emb_weights, open('../full data/CAE/AE_embedding_weights.npy', 'wb'))

# print "Getting embeddings"
# outputs = []

# for i in tqdm(range(0, ICD_data.shape[0], 50)):
# 	ICDbatch, Labbatch = ICD_data[i:i+50], Lab_data[i:i+50]
# 	outputsbatch = model.get_encodings(ICDbatch, Labbatch).data.numpy()
# 	for ob in outputsbatch:
# 		outputs.append(ob)

# pickle.dump(np.array(outputs), open('../full data/CAE/AE_Embeddings', 'wb'))








