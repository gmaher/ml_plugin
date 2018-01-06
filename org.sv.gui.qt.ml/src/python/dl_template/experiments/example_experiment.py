import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from modules import train_utils
from modules import layers as tf_util
from medpy.metric.binary import hd, assd
"""
This file shows an example experiment, it computes a simple threshold

*file reader
*file post processor
*trainer
*saver
*predictor
*logger

The idea is that by keeping these definitions in a separate file
we can make the choice of network configurable by simply referencing
a particular file
"""

class Model(object):
    def __init__(self,global_config,case_config):
        self.global_config = global_config
        self.case_config   = case_config

        self.build_model()

    def train_step(self,Tuple):
        pass

    def save(self):
        pass

    def load(self, model_path=None):
        pass

    def predict(self,xb):
        ypred = xb.copy()
        ypred[ypred>self.threshold]  = 1
        ypred[ypred<=self.threshold] = 0
        return ypred

    def calculate_loss(self,Tuple):
        xb = Tuple[0]
        yb = Tuple[1]
        ypred = xb.copy()
        ypred[ypred>self.threshold]  = 1
        ypred[ypred<=self.threshold] = 0
        return np.sum(np.abs(ypred-yb))

    def build_model(self):
        self.threshold = self.case_config['THRESHOLD']

    def configure_trainer(self):
        pass

def read_file(filename):
    x = np.load(filename+'.X.npy')
    y = np.load(filename+'.Yc.npy')
    return (x,y)

def normalize(Tuple, case_config):
    if case_config['LOCAL_MAX_NORM']:
        x = Tuple[0]
        x = (1.0*x-np.amin(x))/(np.amax(x)-np.amin(x)+1e-5)

        y = Tuple[1]
        y = (1.0*y-np.amin(y))/(np.amax(y)-np.amin(y)+1e-5)
        y = np.round(y).astype(int)
    return (x,y)

def augment(Tuple, global_config, case_config):
    if case_config['ROTATE']:
        Tuple = train_utils.random_rotate(Tuple)

    if case_config['RANDOM_CROP']:
        Tuple = train_utils.random_crop(Tuple,case_config['PATH_PERTURB'],
            global_config['CROP_DIMS'])

    return Tuple

def tuple_to_batch(tuple_list):

    if type(tuple_list) == list and len(tuple_list) == 1:
        tuple_list = tuple_list[0]
    if type(tuple_list) == tuple:
        x = tuple_list[0]
        x = x[np.newaxis,:,:,np.newaxis]
        y = tuple_list[1]
        y = y[np.newaxis,:,:,np.newaxis]
        return x,y
    else:
        x = np.stack([pair[0] for pair in tuple_list])
        x = x[:,:,:,np.newaxis]

        y = np.stack([pair[1] for pair in tuple_list])
        y = y[:,:,:,np.newaxis]
        y = np.round(y)
        return x,y

def calculate_error(ypred,y):
    """assumes ypred and y are thresholded"""
    TP = np.sum(ypred*y)
    FP = np.sum(ypred*(1-y))
    TN = np.sum((1-ypred)*(1-y))
    FN = np.sum((1-ypred)*y)
    HD = hd(y,ypred)
    ASSD = assd(y,ypred)
    DICE = (1.0*TP)/(TP+FN)
    return {"TP":TP, "FP":FP, "TN":TN, "FN":FN, "HD":HD, "ASSD":ASSD, "DICE":DICE}

def evaluate(Tuple,model_instance,config):
    """Note tuple is a single example pair"""
    xb,yb = Tuple
    ypred = model_instance.predict(xb)
    ypred = ypred[0,:,:,0]
    ypred[ypred < config['THRESHOLD']]  = 0
    ypred[ypred >= config['THRESHOLD']] = 1
    ypred = np.round(ypred).astype(int)
    ypred[ypred.shape[0]/2,ypred.shape[0]/2] = 1
    yb = yb[0,:,:,0]
    return calculate_error(ypred,yb),ypred

def log(train_tuple, val_tuple, model_instance, case_config, step):
    batch_dir = case_config['RESULTS_DIR']+'/batch'
    xb,yb = train_tuple
    xv,yv = val_tuple

    ltrain = model_instance.calculate_loss(train_tuple)
    lval   = model_instance.calculate_loss(val_tuple)

    ypred  = model_instance.predict(xv)

    for j in range(xv.shape[0]):

        plt.figure()
        plt.imshow(xv[j,:,:,0],cmap='gray')
        plt.colorbar()
        plt.savefig('{}/{}.{}.x.png'.format(batch_dir,step,j))
        plt.close()

        plt.figure()
        plt.imshow(yv[j,:,:,0],cmap='gray')
        plt.colorbar()
        plt.savefig('{}/{}.{}.y.png'.format(batch_dir,step,j))
        plt.close()

        plt.figure()
        plt.imshow(ypred[j,:,:,0],cmap='gray')
        plt.colorbar()
        plt.savefig('{}/{}.{}.ypred.png'.format(batch_dir,step,j))
        plt.close()

    plt.close('all')
    return ltrain,lval,ypred
