import os

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from itertools import chain
import torchvision.transforms as transforms
import pandas as pd


class DatasetWrapper:
	class __DatasetWrapper:
		"""
		A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
		"""
		def __init__(self, args, cv_iters):
			"""
			create df for features and labels
			remove samples that are not shared between the two tables
			"""
			assert cv_iters > 2, 'Cross validation folds must be more than 2 folds'
			self.cv_iters = cv_iters
			label = ['code']
			phase = ['phase']
			feature = []
			for i in range(1, 91):
				feature.append('Q{}'.format(i))
			df = pd.read_csv('SCL90Cleaned.csv', header = 0, on_bad_lines='skip', usecols = label + phase + feature)
			df = df.dropna()
			
			self.features = df[feature].to_numpy()
			self.labels = df[label].to_numpy()

			self.shuffle()

		def shuffle(self):
			"""
			categorize sample ID by label
			"""
			
			classes = np.unique(self.labels)
			indecies = {}
			for label in classes:
				ind = np.where(self.labels == label)[0][:int(np.count_nonzero(self.labels == label)/self.cv_iters) * self.cv_iters]
				np.random.shuffle(ind)
				indecies[label] = ind.reshape((self.cv_iters, -1))
			
			self.ind = np.empty((self.cv_iters, 0), int)
			for key, value in indecies.items():
				self.ind = np.concatenate((self.ind, value), axis=1)
				
			for i in range(len(self.ind)):
				np.random.shuffle(self.ind[i])
			'''
			self.ind = np.arange(len(self.labels))
			np.random.shuffle(self.ind)
			self.ind = self.ind[:int(len(self.ind)/self.cv_iters) * self.cv_iters].reshape((self.cv_iters, -1))
			'''
			self.CVindex = 1
			self.Testindex = 0


	instance = None
	def __init__(self, args, params, CViters,  shuffle = 0):
		if not DatasetWrapper.instance:
			DatasetWrapper.instance = DatasetWrapper.__DatasetWrapper(args, params.CV_iters)

		if shuffle:
			DatasetWrapper.instance.shuffle()
		DatasetWrapper.Testindex = CViters[0]
		DatasetWrapper.CVindex = CViters[1]

	def __getattr__(self, name):
		return getattr(self.instance, name)

	def features(self, key):
		"""
		Args: 
			key:(string) value from dataset	
		Returns:
			features in list	
		"""
		return DatasetWrapper.instance.features[key]

	def label(self, key):
		"""
		Args: 
			key:(string) the sample key/id	
		Returns:
			label to number 8 or other
		"""
		lab = np.unique(DatasetWrapper.instance.labels)
		ind = np.arange(len(lab))
		ind = ind[np.where(lab == DatasetWrapper.instance.labels[key])[0][0]]
		lab = np.zeros(len(lab))
		lab[ind] = 1
		return lab


	def shuffle(self):
		DatasetWrapper.instance.shuffle()

	def __trainSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of trainning set
		"""

		ind = list(range(DatasetWrapper.instance.cv_iters))
		ind = np.delete(ind, [DatasetWrapper.instance.CVindex, DatasetWrapper.instance.Testindex])

		trainSet = DatasetWrapper.instance.ind[ind].flatten()
		np.random.shuffle(trainSet)
		return trainSet
	
	def __valSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of validation set
		"""

		valSet = DatasetWrapper.instance.ind[DatasetWrapper.instance.CVindex].flatten()
		np.random.shuffle(valSet)
		return valSet

	def __testSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of full dataset
		"""

		testSet = DatasetWrapper.instance.ind[DatasetWrapper.instance.Testindex].flatten()
		np.random.shuffle(testSet)
		return testSet

	def getDataSet(self, dataSetType = 'train'):
		"""
		Args: 
			dataSetType: (string) 'train' or 'val'	
		Returns:
			dataset: (np.ndarray) array of key/id of data set
		"""

		if dataSetType == 'train':
			return self.__trainSet()

		if dataSetType == 'val':
			return self.__valSet()

		if dataSetType == 'test':
			return self.__testSet()

		return self.__testSet()
		


class imageDataset(Dataset):
	"""
	A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
	"""
	def __init__(self, args, dataSetType, params, CViters):
		"""
		initialize DatasetWrapper
		"""
		self.DatasetWrapper = DatasetWrapper(args, params, CViters)

		self.samples = self.DatasetWrapper.getDataSet(dataSetType)

		self.transformer = [
				transforms.Compose([
					transforms.ToTensor()]),  # transform it into a torch tensor
				transforms.Compose([
					transforms.ToTensor()])]

	def __len__(self):
		# return size of dataset
		return len(self.samples)



	def __getitem__(self, idx):
		"""
		Fetch feature and labels from dataset using index of the sample.

		Args:
		    idx: (int) index of the sample

		Returns:
		    feature: (Tensor) feature array
		    label: (int) corresponding label of sample
		"""
		sample = self.samples[idx]
		data = Tensor(self.DatasetWrapper.features(sample).astype(np.uint8))
		
		label = Tensor(self.DatasetWrapper.label(sample).astype(np.uint8))
		return data, label


def fetch_dataloader(args, types, params, CViters):
	"""
	Fetches the DataLoader object for each type in types.

	Args:
	types: (list) has one or more of 'train', 'val'depending on which data is required '' to get the full dataSet
	params: (Params) hyperparameters

	Returns:
	data: (dict) contains the DataLoader object for each type in types
	"""
	dataloaders = {}
	assert CViters[0] != CViters[1], 'ERROR! Test set and validation set cannot be the same!'
	
	if len(types)>0:
		for split in types:
			if split in ['train', 'val', 'test']:
				dl = DataLoader(imageDataset(args, split, params, CViters), 
						batch_size=params.batch_size, 
						shuffle=True,
						num_workers=params.num_workers,
						pin_memory=params.cuda)

				dataloaders[split] = dl
	else:
		dl = DataLoader(imageDataset(args, '', params, CViters), 
				batch_size=params.batch_size, 
				shuffle=True,
				num_workers=params.num_workers,
				pin_memory=params.cuda)

		return dl

	return dataloaders
