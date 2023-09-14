import os
import torch
import logging
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

epslon = 1e-8

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __call__(self):
        return self.avg

def BCE_loss(outputs, labels, weights = (1, 1)):
	'''
	Cross entropy loss with uneven weigth between positive and negative result to manually adjust precision and recall
	'''
	loss = [torch.sum(torch.add(weights[0]*torch.mul(labels[:, i],torch.log(outputs[:, i]+epslon)), weights[1]*torch.mul(1 - labels[:, i],torch.log(1 - outputs[:, i]+epslon)))) for i in range(outputs.shape[1])]
	return -torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def ECE_loss(outputs, labels, weights = (1, 1)):
	'''
	Cross entropy loss with uneven weigth between positive and negative result, add exponential function to positive to manually adjust precision and recall
	'''
	loss = [torch.sum(torch.add(weights[0]*torch.mul(labels[:, i],1.0/(outputs[:, i]+epslon) - 1), weights[1]*torch.mul(1 - labels[:, i],1.0/(1 - outputs[:, i]+epslon)-1))) for i in range(outputs.shape[1])]
	return torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def F_ECE_loss(outputs, labels, gamma = 2):
	'''
	Cross entropy loss with uneven weigth between positive and negative result, add exponential function to positive to manually adjust precision and recall
	'''
	loss = [torch.sum(torch.add(	torch.mul(labels[:, i],1.0/(outputs[:, i]+epslon) - 1), 
					-torch.mul(torch.pow(outputs[:, i], gamma), torch.mul(1 - labels[:, i], torch.log(1 - outputs[:, i]+epslon))))) for i in range(outputs.shape[1])]
	return torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def Focal_loss(outputs, labels, gamma = 2):
	loss = [torch.sum(torch.add(	torch.mul(torch.pow(1 - outputs[:, i], gamma), torch.mul(labels[:, i], torch.log(outputs[:, i]+epslon))), 
					torch.mul(torch.pow(outputs[:, i], gamma), torch.mul(1 - labels[:, i], torch.log(1 - outputs[:, i]+epslon))))) for i in range(outputs.shape[1])]
	return -torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def get_loss(loss_name):
	loss_name = loss_name.lower()
	if loss_name == 'bce':
		return BCE_loss
	elif loss_name == 'ece': 
        	return ECE_loss
	elif loss_name == 'focal': 
        	return Focal_loss
	elif loss_name == 'f_ece': 
        	return F_ECE_loss
	else:
		logging.warning("No loss function with the name {} found, please check your spelling.".format(loss_name))
		logging.warning("loss function List:")
		logging.warning("    BCE")
		logging.warning("    ECE")
		logging.warning("    focal")
		logging.warning("    F_ECE")
		import sys
		sys.exit()
		
def get_default_gamma(loss_name):
	loss_name = loss_name.lower()
	if loss_name == 'bce':
		return 1
	elif loss_name == 'ece': 
        	return 1
	elif loss_name == 'focal': 
        	return 2
	elif loss_name == 'f_ece': 
        	return 2
	else:
		logging.warning("No loss function with the name {} found, please check your spelling.".format(loss_name))
		logging.warning("loss function List:")
		logging.warning("    BCE")
		logging.warning("    ECE")
		logging.warning("    focal")
		logging.warning("    F_ECE")
		import sys
		sys.exit()
		
def get_AUC(outputs):
	AUC = []
	for i in range(outputs[0].shape[1]):
		fpr, tpr, thresholds = metrics.roc_curve(outputs[1][:, i], outputs[0][:, i], pos_label=1)
		AUC.append(metrics.auc(fpr, tpr))
	return np.mean(AUC)
	
def get_eval_multi(outputs):
	result = defaultdict(list)

	for i in range(outputs[0].shape[1]):
		fpr, tpr, thresholds = metrics.roc_curve(outputs[1][:, i], outputs[0][:, i], pos_label=1)
		result['AUC'].append(metrics.auc(fpr, tpr))
		precision, recall, thresholds = metrics.precision_recall_curve(outputs[1][:, i], outputs[0][:, i])
		f1_scores = 2*recall*precision/(recall+precision + 1e-17)
		ind = np.argmax(f1_scores)
		
		outputs[0][:, i] = outputs[0][:, i] > thresholds[ind]
		
		result['threshold'].append(thresholds[ind])
		result['acc'].append(metrics.accuracy_score(outputs[1][:, i], outputs[0][:, i]))
		result['Precision'].append(metrics.precision_score(outputs[1][:, i], outputs[0][:, i], average='weighted', zero_division = 0))
		result['Recall'].append(metrics.recall_score(outputs[1][:, i], outputs[0][:, i], average='weighted'))
		result['F0.5'].append(metrics.fbeta_score(outputs[1][:, i], outputs[0][:, i], average='weighted', beta=0.5))
		result['F0'].append(metrics.fbeta_score(outputs[1][:, i], outputs[0][:, i], average='weighted', beta=0))
		result['F1'].append(metrics.f1_score(outputs[1][:, i], outputs[0][:, i], average='weighted'))
	
	return result
	

def plot_AUC_SD(netlist, evalmatices):
	plt.clf()
	possitive_ratio = np.loadtxt("./data/possitive_ratio.txt", dtype=float)
	logging.warning('    Creating standard diviation image for {}'.format('-'.join(netlist)))
	png_file = 'Crossvalidation_Analysis_{}.tex'.format('-'.join(netlist))

	if len(netlist) == 0:
		return


	plt.clf()
	fig, ax = plt.subplots(2)
	fig.suptitle('Accruacy, F1 for {}'.format('-'.join(netlist)))
	
	data = []
	for net in netlist:
		data.append(np.array(evalmatices[net]).T[0])

	ax[0].boxplot(data, showfliers=False)
	ax[0].set_ylabel('Accruacy')

	data = []
	for net in netlist:
		data.append(np.array(evalmatices[net]).T[1])

	ax[1].boxplot(data, showfliers=False)
	ax[1].set_ylabel('F1')
	ax[1].set_xticklabels(netlist, fontsize=10)
	
	import tikzplotlib
	
	logging.warning('    Saving standard diviation image for {} \n'.format('-'.join(netlist)))
	tikzplotlib.save(png_file)

	
