import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from modules import train_utils
from modules import layers as tf_util
#from medpy.metric.binary import hd, assd
from modules import io
from modules import vascular_data as sv
"""
This file builds the I2INet network and sets up the required

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

        self.configure_trainer()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train_step(self,Tuple):
        self.global_step = self.global_step+1
        xb,yb = Tuple

        if np.sum(np.isnan(xb)) > 0: return
        if np.sum(np.isnan(yb)) > 0: return

        self.sess.run(self.train,{self.x:xb,self.y:yb})

    def save(self):
        model_dir  = self.case_config['MODEL_DIR']
        model_name = self.case_config['MODEL_NAME']
        self.saver.save(
            self.sess,model_dir+'/{}'.format(model_name))

    def load(self, model_path=None):
        if model_path == None:
            model_dir  = self.case_config['MODEL_DIR']
            model_name = self.case_config['MODEL_NAME']
            model_path = model_dir + '/' + model_name
        self.saver.restore(self.sess, model_path)

    def predict(self,xb):
        return self.sess.run(self.yclass,{self.x:xb})

    def calculate_loss(self,Tuple):
        xb = Tuple[0]
        yb = Tuple[1]
        return self.sess.run(self.loss,{self.x:xb,self.y:yb})

    def build_model(self):
        CROP_DIMS   = self.global_config['CROP_DIMS']
        C           = self.case_config['NUM_CHANNELS']
        LEAK        = self.global_config['LEAK']
        NUM_FILTERS = self.global_config['NUM_FILTERS']
        LAMBDA      = self.global_config['L2_REG']
        INIT        = self.global_config['INIT']

        leaky_relu = tf.contrib.keras.layers.LeakyReLU(LEAK)

        self.x = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)
        self.y = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)

        #I2INetFC
        # self.yclass,self.yhat, self.i2i_yclass, self.i2i_yhat =\
        #  tf_util.I2INetFC(self.x, nfilters=NUM_FILTERS, activation=leaky_relu, init=INIT)

        self.yclass,self.yhat,_,_ = tf_util.I2INet(self.x,nfilters=NUM_FILTERS,
            activation=leaky_relu,init=INIT)

        #Loss
        # self.loss = tf.reduce_mean(
        #        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.i2i_yhat,labels=self.y))

        self.loss = tf.reduce_mean(
               tf.nn.sigmoid_cross_entropy_with_logits(logits=self.yhat,labels=self.y))

        self.loss = self.loss + tf_util.l2_reg(LAMBDA)

        self.saver = tf.train.Saver()

    def configure_trainer(self):
        LEARNING_RATE = self.global_config["LEARNING_RATE"]
        self.global_step = tf.Variable(0, trainable=False)
        boundaries = [5000, 10000, 15000]
        values = [LEARNING_RATE, LEARNING_RATE/10, LEARNING_RATE/100, LEARNING_RATE/1000]
        learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

        self.opt = tf.train.AdamOptimizer(learning_rate)
        self.train = self.opt.minimize(self.loss)

def read_file(filename):
    x  = np.load(filename+'.X.npy')
    y  = np.load(filename+'.Y.npy')
    yc = np.load(filename+'.Yc.npy')
    return (x,y,yc,filename)

def normalize(Tuple, case_config):
    y = Tuple[1]
    y = (1.0*y-np.amin(y))/(np.amax(y)-np.amin(y)+1e-5)
    y = np.round(y).astype(int)

    x = Tuple[0]

    if case_config['NORMALIZE'] == "LOCAL_NORM":
        x = Tuple[0]
	x = (1.0*x-np.mean(x))/(np.std(x)+1e-5)

    elif case_config['NORMALIZE'] == "LOCAL_MAX_NORM":
	x = Tuple[0]
	x = (1.0*x-np.amin(x))/(np.amax(x)-np.amin(x)+1e-5)

    elif case_config['NORMALIZE'] == "WINDOW":
        # fn = Tuple[2]
        # remove = '/'+fn.split('/')[-2]+'/'+fn.split('/')[-1]
        # fn = fn.replace(remove,'')
        # fn = fn+'/image_stats.csv'
        # stat_dict = io.read_csv(fn)
        blood_mean = np.mean(x[y>0.1])
        blood_std  = np.std(x[y>0.1])
        x = sv.window_image(x,float(blood_mean)*0.7,float(blood_std)*6)

    else:
        raise RuntimeError("Unsupported normalization type".format(case_config['NORMALIZE']))

    return (x,y)

def augment(Tuple, global_config, case_config):
    Tuple_ = (Tuple[0],Tuple[1])
    if case_config['ROTATE']:
        Tuple_ = train_utils.random_rotate(Tuple_)

    if case_config['RANDOM_CROP']:
        Tuple_ = train_utils.random_crop(Tuple_,case_config['PATH_PERTURB'],
            global_config['CROP_DIMS'])

    x,y = Tuple_

    return (x,y)

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
    xb,yb,yc = Tuple
    ypred = model_instance.predict(xb)
    ypred = ypred[0,:,:,0]
    ypred[ypred < config['THRESHOLD']]  = 0
    ypred[ypred >= config['THRESHOLD']] = 1
    ypred = np.round(ypred).astype(int)
    ypred[ypred.shape[0]/2,ypred.shape[0]/2] = 1
    yb = yb[0,:,:,0]
    err_dict = calculate_error(ypred,yb)
    err_dict['RADIUS'] = np.sqrt((1.0*np.sum(yc))/np.pi)
    return err_dict, ypred

def log(train_tuple, val_tuple, model_instance, case_config, step):
    batch_dir = case_config['RESULTS_DIR']+'/batch'
    xb,yb = train_tuple
    xv,yv = val_tuple

    ltrain = model_instance.calculate_loss(train_tuple)
    lval   = model_instance.calculate_loss(val_tuple)

    ypred  = model_instance.predict(xv)

    # yi2i   = model_instance.sess.run(model_instance.i2i_yclass,{model_instance.x:xv})

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

        # plt.figure()
        # plt.imshow(yi2i[j,:,:,0],cmap='gray')
        # plt.colorbar()
        # plt.savefig('{}/{}.{}.yi2i.png'.format(batch_dir,step,j))
        # plt.close()

    plt.close('all')
    return ltrain,lval,ypred
