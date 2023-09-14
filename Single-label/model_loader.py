import os
import sys
import torch
import logging
import torch.nn as nn
def loadModel(hyperparams, netname, channels):
    Netpath = 'Model'
    Netfile = os.path.join(Netpath, netname)
    Netfile = os.path.join(Netfile, netname + '.py')
    assert os.path.isfile(Netfile), "No python file found for {}, (file name is case sensitive)".format(Netfile)
    netname = netname.lower()
    if netname == 'layer1':
        return loadlayer1(hyperparams, channels)
    elif netname == 'layer2': 
        return loadlayer2(hyperparams, channels)
    elif netname == 'layer3': 
        return loadlayer3(hyperparams, channels)
    elif netname == 'layer4': 
        return loadlayer4(hyperparams, channels)
    else:
        logging.warning("No model with the name {} found, please check your spelling.".format(netname))

        sys.exit()

def get_model_list(netname = ''):
    netname = netname.lower()
    if netname == '':
        return ['layer1', 'layer2', 'layer3', 'layer4']

    if netname in ['layer1', 'layer2', 'layer3', 'layer4']:
        return [netname]

    logging.warning("No model with the name {} found, please check your spelling.".format(netname))
    logging.warning("Net List:")
    logging.warning("    layer1")
    logging.warning("    layer2")
    logging.warning("    layer3")
    logging.warning("    layer4")
    sys.exit()
    
def loadlayer1(hyperparams, channels):
    from Model.layer1.layer1 import layer1
    logging.warning("Loading layer1 Model")
    return layer1(hyperparams, channels)

def loadlayer2(hyperparams, channels):
    from Model.layer2.layer2 import layer2
    logging.warning("Loading layer2 Model")
    return layer2(hyperparams, channels)
    
def loadlayer3(hyperparams, channels):
    from Model.layer3.layer3 import layer3
    logging.warning("Loading layer3 Model")
    return layer3(hyperparams, channels)
    
def loadlayer4(hyperparams, channels):
    from Model.layer4.layer4 import layer4
    logging.warning("Loading layer4 Model")
    return layer4( hyperparams, channels)

def weight_ini(m):
    torch.manual_seed(230)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
    

