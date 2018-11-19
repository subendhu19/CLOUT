import numpy as np
import pickle
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

vocabsize_icd = 942
vocabsize_meds = 3202
vocabsize_labs = 284 #all 681
vocabsize = vocabsize_icd+vocabsize_meds+vocabsize_labs

input_seqs_icd = np.array(pickle.load(open('../full data/MIMICIIIPROCESSED.3digitICD9.seqs')))
input_seqs_meds = np.array(pickle.load(open('../full data/MIMICIIIPROCESSED.meds.seqs')))
input_seqs_labs = np.array(pickle.load(open('../full data/MIMICIIIPROCESSED.abnlabs.seqs')))
input_seqs_fullicd = np.array(pickle.load(open('../full data/MIMICIIIPROCESSED.seqs')))

labels = np.array(pickle.load(open('../full data/MIMICIIIPROCESSED.morts')))

# fout = open("logistic_regression_interpretations.txt", 'w')

def combine_encounter(seqs, length):
	ret_vector = np.zeros(length)
	for enc in seqs:
		for code in enc:
			ret_vector[code] = 1

	return ret_vector

input_seqs = np.array([np.concatenate((combine_encounter(input_seqs_icd[i], vocabsize_icd), combine_encounter(input_seqs_meds[i], vocabsize_meds), combine_encounter(input_seqs_labs[i], vocabsize_labs)), axis=0) for i in range(0, len(input_seqs_icd))])

trainratio = 0.7
validratio = 0.1
testratio = 0.2

trainlindex = int(len(input_seqs_icd)*trainratio)
validlindex = int(len(input_seqs_icd)*(trainratio + validratio))

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

icditems = pickle.load(open('../full data/type dictionaries/MIMICIIIPROCESSED.types', 'rb'))
meditems = pickle.load(open('../full data/type dictionaries/MIMICIIIPROCESSED.meds.types', 'rb'))
labitems = pickle.load(open('../full data/type dictionaries/MIMICIIIPROCESSED.abnlabs.types', 'rb'))

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

def get_factor(i, patid):
	if i<942:
		return get_ICD(pattofullicd_dict[patid][i]), 0
	elif i<942+3202:
		return get_med(i-942), 1
	else:
		return get_lab(i-3202-942), 2

best_aucrocs = []
for run in range(10):
	print 'Run', run

	perm = np.random.permutation(input_seqs.shape[0])
	rinput_seqs = input_seqs#[perm]
	rinput_seqs_fullicd = input_seqs_fullicd#[perm]
	rlabels = labels#[perm]
	r_input_icd = input_seqs_icd#[perm]

	train_input_seqs = rinput_seqs[:trainlindex]
	train_input_seqs_fullicd = rinput_seqs_fullicd[:trainlindex]
	train_labels = rlabels[:trainlindex]

	valid_input_seqs = rinput_seqs[trainlindex:validlindex]
	valid_input_seqs_fullicd = rinput_seqs_fullicd[trainlindex: validlindex]
	valid_labels = rlabels[trainlindex:validlindex]

	test_input_seqs = rinput_seqs[validlindex:]
	test_input_seqs_fullicd = rinput_seqs_fullicd[validlindex:]
	test_labels = rlabels[validlindex:]
	test_input_seqs_interpretations = r_input_icd[validlindex:]

	model = linear_model.LogisticRegression()

	model.fit(train_input_seqs, train_labels)

	vpredict_probabilities = np.array([a[1] for a in model.predict_proba(valid_input_seqs)])
	print "Validation AUC_ROC: ", roc_auc_score(valid_labels, vpredict_probabilities)

	predict_probabilities = np.array([a[1] for a in model.predict_proba(test_input_seqs)])
	print "Test AUC_ROC: ", roc_auc_score(test_labels, predict_probabilities)

	best_aucrocs.append(roc_auc_score(test_labels, predict_probabilities))

print "Average AUCROC:", np.mean(best_aucrocs), "+/-", np.std(best_aucrocs) 


interpretation_file = open("Log_Reg_Interpretations.txt", 'w')

pattofullicd_dict = {}
for i in range(len(test_input_seqs_interpretations)):
	icdtofullicd_dict = {}
	for j in range(len(test_input_seqs_interpretations[i])):
		for k in range(len(test_input_seqs_interpretations[i][j])):
			icdtofullicd_dict[test_input_seqs_interpretations[i][j][k]] = test_input_seqs_fullicd[i][j][k]
	pattofullicd_dict[i] = icdtofullicd_dict

coeffs = np.array(model.coef_[0])
for patid in range(len(test_input_seqs)):
	test_input = test_input_seqs[patid]

	scores = (test_input*coeffs)
	# scores = coeffs
	risk_scores = []
	for i in range(len(scores)):
		if test_input[i]>0:
			factors = get_factor(i, patid)
			risk_scores.append((factors[0], scores[i]))
	risk_scores.sort(key=lambda tup: tup[1], reverse=True)

	top_risk_factors = risk_scores[:10]

	if (predict_probabilities[patid] > 0.5):
		interpretation_file.write("ID: " + str(patid) + " True label: "+str(test_labels[patid])+"\n")
		for rf in top_risk_factors:
			interpretation_file.write(str(rf)+"\n")
		interpretation_file.write("\n")

interpretation_file.close()
			

# fpr, tpr, _ = roc_curve(test_labels, predict_probabilities)
# pickle.dump({"FPR":fpr, "TPR":tpr}, open('roc_lr.p', 'wb'))
# actual_predictions = (predict_probabilities>0.5)*1

# print classification_report(test_labels, actual_predictions)
# coeffs = np.array(model.coef_[0])

# icd_scores = {}
# icd_totals = {}
# med_scores = {}
# med_totals = {}
# lab_scores = {}
# lab_totals = {}

# for patid in range(len(test_input_seqs)):
# 	test_input = test_input_seqs[patid]
# 	scores = (test_input*coeffs)
# 	# scores = coeffs
# 	for i in range(len(scores)):
# 		if test_input[i]>0:
# 			factors = get_factor(i, patid)
# 			if factors[1] == 0:
# 				if factors[0] in icd_scores:
# 					icd_scores[factors[0]] += scores[i]
# 					icd_totals[factors[0]] += 1
# 				else:
# 					icd_scores[factors[0]] = scores[i]
# 					icd_totals[factors[0]] = 1
# 			elif factors[1] == 1:
# 				if factors[0] in med_scores:
# 					med_scores[factors[0]] += scores[i]
# 					med_totals[factors[0]] += 1
# 				else:
# 					med_scores[factors[0]] = scores[i]
# 					med_totals[factors[0]] = 1
# 			else:
# 				if factors[0] in lab_scores:
# 					lab_scores[factors[0]] += scores[i]
# 					lab_totals[factors[0]] += 1
# 				else:
# 					lab_scores[factors[0]] = scores[i]
# 					lab_totals[factors[0]] = 1


# icd_averages = []
# med_averages = []
# lab_averages = []

# for factor in icd_scores:
# 	icd_averages.append((factor, icd_scores[factor]/icd_totals[factor]))
# icd_averages.sort(key=lambda tup: tup[1], reverse=True)
# fout.write("ICD codes:\n")
# for item in icd_averages:
# 	fout.write(item[0]+"-"+str(item[1])+"\n")
# fout.write("\n")

# for factor in med_scores:
# 	med_averages.append((factor, med_scores[factor]/med_totals[factor]))
# med_averages.sort(key=lambda tup: tup[1], reverse=True)
# fout.write("Medications:\n")
# for item in med_averages:
# 	fout.write(item[0]+"-"+str(item[1])+"\n")
# fout.write("\n")

# for factor in lab_scores:
# 	lab_averages.append((factor, lab_scores[factor]/lab_totals[factor]))
# lab_averages.sort(key=lambda tup: tup[1], reverse=True)
# fout.write("Lab components:\n")
# for item in lab_averages:
# 	fout.write(item[0]+"-"+str(item[1])+"\n")
# fout.write("\n")

# scores = [(scores[i], get_factor(i)) for i in range(len(scores)) if test_input[i]>0]
# scores.sort(key=lambda tup: tup[0], reverse=True)

# for factor in scores:
# 	if factor[1] in ["\"Encephalopathy NOS\"", "\"Bleed esoph var oth dis\"", "\"Lactulose Enema\"", "\"Cirrhosis of liver NOS\"", "\"Urin tract infection NOS\"", "\"Phytonadione\"", "\"Hy kid NOS w cr kid I-IV\"", "\"Mal neo liver", "\"Red blood cells\"", "\"RDW\"", "\"Hemoglobin\"", "\"0.9% Sodium Chloride\""]:
# 		print factor

#for factor in scores:


# fout.close()


