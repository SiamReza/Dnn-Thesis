import torch.nn as nn


class layer4(nn.Module):
	def __init__(self, hyperparams, channels):
		"""
		Args:
		    params: (Params) contains num_channels
		"""
		super(layer4, self).__init__()
		self.fc = nn.Sequential(
				nn.Dropout(hyperparams.dropout_rate), 
				nn.Linear(90, hyperparams.hidden_layer1), 
				nn.ReLU(),
				
				nn.Dropout(hyperparams.dropout_rate), 
				nn.Linear(hyperparams.hidden_layer1, hyperparams.hidden_layer2), 
				nn.ReLU(),
				
				nn.Dropout(hyperparams.dropout_rate), 
				nn.Linear(hyperparams.hidden_layer2, hyperparams.hidden_layer3), 
				nn.ReLU(),
				
				nn.Dropout(hyperparams.dropout_rate), 
				nn.Linear(hyperparams.hidden_layer3, channels),
				nn.Softmax(dim=1)
				)

	def forward(self, x):
		"""
		This function defines how we use the components of our network to operate on an input batch.

		Args:
		    X: (Variable) features.

		Returns:
		    out: (Variable) dimension batch_size x 1 with the log probabilities for the prediction.

		Note: the dimensions after each step are provided
		"""
		x = x/4.0
		x = x.flatten(1)
		
		return self.fc(x)

