from numpy import isnan, inf, savetxt
import torch
import logging
import gc

import torch.nn as nn
import torch.backends.cudnn as cudnn
from Evaluation_Matix import *
from utils import *
import model_loader
from data_loader import fetch_dataloader
from tqdm import tqdm
from datetime import datetime

class Solver:
	def __init__(self, args, params, CViter):
		def init_weights(m):
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				m.bias.data.fill_(0.01)
		torch.cuda.empty_cache() 
		self.args = args
		if not args.gamma:
			get_default_gamma(args.loss)
		self.params = params
		self.CViter = CViter
		self.dataloaders = fetch_dataloader(args, ['train', 'val', 'test'], params, CViter) 
		self.model = model_loader.loadModel(params.hyperparam, netname = args.network, channels = params.channels).cuda()
		self.model.apply(init_weights)
		self.optimizer = torch.optim.Adam(	self.model.parameters(), 
							params.hyperparam.learning_rate, 
							betas=(0.9, 0.999), 
							eps=1e-08, 
							weight_decay = params.weight_decay, 
							amsgrad=False)
		self.loss_fn = nn.BCELoss() #get_loss(args.loss)
		
		
	def __step__(self):
		torch.cuda.empty_cache() 
		logging.info("Training")
		losses = AverageMeter()
		# switch to train mode
		self.model.train()
		loss = []
		print('**********TRAINING**********')
		with tqdm(total=len(self.dataloaders['train'])) as t:
			for i, (datas, label) in enumerate(self.dataloaders['train']):
				logging.info("        Loading Varable")
				# compute output
				logging.info("        Compute output")
				output = self.model(torch.autograd.Variable(datas.cuda())).double()

				# measure record cost
				cost = self.loss_fn(output, torch.autograd.Variable(label.cuda()).double())
				assert not isnan(cost.cpu().data.numpy().any()),  "Gradient exploding, Loss = {}".format(cost.cpu().data.numpy())
				losses.update(cost.cpu().data.numpy(), len(datas))
				
				del output

				# compute gradient and do SGD step
				logging.info("        Compute gradient and do SGD step")
				self.optimizer.zero_grad()
				cost.backward()
				self.optimizer.step()
			
				gc.collect()
				t.set_postfix(loss='{:05.3f}'.format(losses()))
				t.update()

		return loss
	
	
	def validate(self, dataset_type = 'val'):
		torch.cuda.empty_cache() 
		logging.info("Validating")
		losses = AverageMeter()
		if dataset_type =='test':
			self.__resume_checkpoint__('best')
		outputs = [np.empty((0, 5), float), np.empty((0, 5), float)]
		# switch to evaluate mode
		self.model.eval()
		print('----------VALIDATING----------')
		with tqdm(total=len(self.dataloaders['val'])) as t:
			for i, (datas, label) in (enumerate(self.dataloaders[dataset_type])):
				logging.info("        Compute output")
				output = self.model(torch.autograd.Variable(datas.cuda())).double()
				label_var = torch.autograd.Variable(label.cuda()).double()
				outputs[0] = np.concatenate((outputs[0], output.cpu().data.numpy()), axis=0)
				outputs[1] = np.concatenate((outputs[1], label_var.cpu().data.numpy()), axis=0)
				logging.info("        Computing loss")
				loss = self.loss_fn(output, torch.autograd.Variable(label.cuda()).double())
				assert not isnan(loss.cpu().data.numpy()),  "Overshot loss, Loss = {}".format(loss.cpu().data.numpy())
				
				# measure record cost
				losses.update(loss.cpu().data.numpy(), len(datas))
				
				del output
				del label_var
				
				gc.collect()
				t.update()
		
		return outputs


	def train(self):
		start_epoch = 0
		best_AUC = 0	

		if self.args.resume:
			logging.warning('Resuming Checkpoint')
			start_epoch, best_AUC = self.__resume_checkpoint__('')
			if not start_epoch < self.params.epochs:
				logging.warning('Skipping training for finished model\n')
				return []			
			
		logging.warning('    Starting With Best AUC = {AUC:.4f}'.format(AUC = best_AUC))
		logging.warning('Initialize training from {} to {} epochs'.format(start_epoch, self.params.epochs))

		for epoch in range(start_epoch, self.params.epochs):
			logging.warning('CV [{}], Training Epoch: [{}/{}]'.format('_'.join(tuple(map(str, self.CViter))), epoch+1, self.params.epochs))

			self.__step__()
			gc.collect()

			# evaluate on validation set
			val_AUC = get_AUC(self.validate())
			gc.collect()

			# remember best AUC and save checkpoint
			logging.warning('    AUC {AUC:.4f};\n'.format(AUC = val_AUC))		
			if val_AUC > best_AUC:
				self.__save_checkpoint__({
					'epoch': epoch + 1,
					'state_dict': self.model.state_dict(),
					'AUC': val_AUC,
					'optimizer' : self.optimizer.state_dict(),
					}, 'best')
				best_AUC = val_AUC
				logging.warning('    Saved Best AUC model with  \n{} \n'.format(val_AUC))

			self.__save_checkpoint__({
					'epoch': epoch + 1,
					'state_dict': self.model.state_dict(),
					'AUC': best_AUC,
					'optimizer' : self.optimizer.state_dict(),
					}, '')
					
			if epoch % 5 == 4 and epoch > 0:
				print('*****************decay**************\n')
				
				self.__learning_rate_decay__(self.optimizer, self.params.hyperparam.lrDecay)

		gc.collect()
		logging.warning('Training finalized with best average AUC {}\n'.format(best_AUC))
		return
		
	def __save_checkpoint__(self, state, checkpoint_type):
		checkpointpath, checkpointfile = get_checkpointname(	self.args, 
									checkpoint_type, 
									self.CViter)
		if not os.path.isdir(checkpointpath):
			os.mkdir(checkpointpath)
			
		torch.save(state, checkpointfile)


	def __resume_checkpoint__(self, checkpoint_type):
		_, checkpointfile = get_checkpointname(self.args, checkpoint_type, self.CViter)
		
		if not os.path.isfile(checkpointfile):
			return 0, 0
		else:
			logging.info("Loading checkpoint {}".format(checkpointfile))
			checkpoint = torch.load(checkpointfile)
			start_epoch = checkpoint['epoch']
			AUC = checkpoint['AUC']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			
				
			return start_epoch, AUC

	def __learning_rate_decay__(self, optimizer, decay_rate):
		if decay_rate < 1:
			for param_group in optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * decay_rate
				print('    learning rate {}'.format(param_group['lr']))
