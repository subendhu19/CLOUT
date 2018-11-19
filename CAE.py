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

vocabsize_icd = 942
vocabsize_meds = 3202
vocabsize_labs = 284 #all 681 284

class CAE(nn.Module):
	def __init__(self, epochs=5, batchsize=50, embsize=100, lamb=0.01):
		super(CAE, self).__init__()
		self.epochs = epochs
		self.batchsize = batchsize
		self.embsize = embsize
		self.lamb = lamb

		self.emb = nn.Linear(vocabsize_icd + vocabsize_meds + vocabsize_labs, self.embsize)

		self.out = nn.Linear(self.embsize, vocabsize_icd + vocabsize_meds + vocabsize_labs)

		self.reconloss = nn.MSELoss(size_average=True)

	def forward(self, input_icd, input_med, input_lab):

		input_full = torch.cat((input_icd, input_med, input_lab),1)
		input_onlyicd = torch.cat((input_icd, Variable(torch.zeros(input_med.size(0), input_med.size(1)).float()), Variable(torch.zeros(input_lab.size(0), input_lab.size(1)).float())), 1)
		input_onlymed = torch.cat((Variable(torch.zeros(input_icd.size(0), input_icd.size(1)).float()), input_med, Variable(torch.zeros(input_lab.size(0), input_lab.size(1)).float())), 1)
		input_onlylab = torch.cat((Variable(torch.zeros(input_icd.size(0), input_icd.size(1)).float()), Variable(torch.zeros(input_med.size(0), input_med.size(1)).float()), input_lab), 1)

		hidden_full = F.relu(self.emb(input_full))
		hidden_onlyicd = F.relu(self.emb(input_onlyicd))
		hidden_onlymed = F.relu(self.emb(input_onlymed))	
		hidden_onlylab = F.relu(self.emb(input_onlylab))		 

		output_full = F.relu(self.out(hidden_full))
		output_onlyicd = F.relu(self.out(hidden_onlyicd))
		output_onlymed = F.relu(self.out(hidden_onlymed))
		output_onlylab = F.relu(self.out(hidden_onlylab))	

		return [output_full, output_onlyicd, output_onlymed, output_onlylab, hidden_onlyicd, hidden_onlymed, hidden_onlylab, hidden_full]

	def get_encodings(self, ICD_data, Med_data, Lab_data):
		return self.forward(Variable(torch.from_numpy(ICD_data).float()), Variable(torch.from_numpy(Med_data).float()), Variable(torch.from_numpy(Lab_data).float()))[-1]

	def correlation_coef(self, x, y):
		vx = x - torch.mean(x)
		vy = y - torch.mean(y)

		cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
		return cost

	def joint_cumulant_by_var(self, x, y, z):	
		vx = x - torch.mean(x)
		vy = y - torch.mean(y)
		vz = z - torch.mean(z)

		cost = torch.sum(vx * vy * vz) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vz ** 2)))
		return cost

		# e_xyz = torch.mean(x * y * z)
		# e_xy = torch.mean(x * y)
		# e_yz = torch.mean(y * z)
		# e_xz = torch.mean(x * z)
		# e_x = torch.mean(x)
		# e_y = torch.mean(y)
		# e_z = torch.mean(z)

		# kappa = e_xyz - (e_xy * e_z) - (e_xz * e_y) - (e_yz * e_x) + (2*e_x*e_y*e_z) 


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

				loss_recon = self.reconloss(outputs[0], torch.cat((ICDbatchvar, Medbatchvar, Labbatchvar),1)) + self.reconloss(outputs[1], torch.cat((ICDbatchvar, Medbatchvar, Labbatchvar),1)) \
						+ self.reconloss(outputs[2], torch.cat((ICDbatchvar, Medbatchvar, Labbatchvar),1)) + self.reconloss(outputs[3], torch.cat((ICDbatchvar, Medbatchvar, Labbatchvar),1))

				loss_cr = self.joint_cumulant_by_var(outputs[4], outputs[5], outputs[6])

				loss = loss_recon - (self.lamb*loss_cr)

				losses.append(loss.data[0])

				optimizer.zero_grad()
			    
				loss.backward()
			    
				optimizer.step()
				# print 'recon loss:', loss_recon.data[0], 'loss_cr:', loss_cr.data[0]

			print 'Epoch loss:', np.mean(losses)

			if abs(np.mean(losses) - prev_loss) < 0.00005:
				break

			prev_loss = np.mean(losses)


model = CAE(10,50,175,0.01)
ICD_data = pickle.load(open('../full data/CAE/CAEEntries.3digitICD9','r'))
Med_data = pickle.load(open('../full data/CAE/CAEEntries.meds','r'))
Lab_data = pickle.load(open('../full data/CAE/CAEEntries.abnlabs','r'))
model.fit(ICD_data, Med_data, Lab_data)

emb_weights = model._modules['emb'].weight.data.numpy().T
print 'Pickled embedding weights. Shape:', np.array(emb_weights).shape
pickle.dump(emb_weights, open('../full data/CAE/CAE_embedding_weights.npy', 'wb'))

# outputs = model.get_encodings(ICD_data, Med_data, Lab_data)
# print np.array(outputs).shape
# pickle.dump(outputs, open('../full data/CAE/Embeddings', 'wb'))








