import tensorflow as tf
import numpy as np
import vtk
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('vtkfolder')

args = parser.parse_args()

folder = os.path.abspath(args.vtkfolder)

#################################
# Define Parameters
#################################
C=1
crop_dims = 128
Nbatch = 32
init = 6e-2
Nfilters = 32
EPS=1e-4
leaky_relu = tf.contrib.keras.layers.LeakyReLU(0.2)

#################################
# Build tensorflow model
#################################
x = tf.placeholder(shape=[None,crop_dims,crop_dims,C],dtype=tf.float32)
y = tf.placeholder(shape=[None,crop_dims,crop_dims,C],dtype=tf.float32)

yclass,yhat,o3,o4 = tf_util.I2INet(x,nfilters=Nfilters,activation=leaky_relu,init=init)

y_vec = tf.reshape(yclass, (Nbatch,crop_dims**2))

sp = tf_util.fullyConnected(y_vec,crop_dims,leaky_relu, std=init, scope='sp1')
sp = tf_util.fullyConnected(y_vec,crop_dims**2,leaky_relu, std=init, scope='sp2')
sp = tf.reshape(sp, (Nbatch,crop_dims,crop_dims,1))

y_sp = tf_util.conv2D(sp, nfilters=Nfilters, activation=leaky_relu,init=init, scope='sp3')
y_sp_1 = tf_util.conv2D(y_sp, nfilters=Nfilters, activation=leaky_relu, init=init,scope='sp4')
y_sp_2 = tf_util.conv2D(y_sp_1, nfilters=Nfilters, activation=leaky_relu, init=init,scope='sp5')

yhat = tf_util.conv2D(y_sp_2, nfilters=1, activation=tf.identity, init=init,scope='sp6')
yclass = tf.sigmoid(yhat)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#################################
# Load model
#################################
saver = tf.train.Saver()
saver.restore(sess,'./models/i2i_CT/i2i_CT')

#################################
# Get vts files
#################################
vts_files = os.listdir(folder)
