import sys
sys.path += ['', '/home/marsdenlab/anaconda2/lib/python2.7/site-packages/google/','/home/marsdenlab/anaconda2/lib/python2.7/site-packages/google/protobuf','/home/marsdenlab/libraries/vmtk/build/Install/lib/python2.7/site-packages', '/home/marsdenlab/projects/SV3', '/home/marsdenlab/anaconda2/lib/python27.zip', '/home/marsdenlab/anaconda2/lib/python2.7', '/home/marsdenlab/anaconda2/lib/python2.7/plat-linux2', '/home/marsdenlab/anaconda2/lib/python2.7/lib-tk', '/home/marsdenlab/anaconda2/lib/python2.7/lib-old', '/home/marsdenlab/anaconda2/lib/python2.7/lib-dynload', '/home/marsdenlab/anaconda2/lib/python2.7/site-packages', '/home/marsdenlab/anaconda2/lib/python2.7/site-packages/Sphinx-1.5.4-py2.7.egg', '/home/marsdenlab/libraries/pybedtools', '/home/marsdenlab/projects/tcl_code/python/src/pyemd', '/home/marsdenlab/anaconda2/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg']
sys.path = [s for s in sys.path if "Externals-build" not in s]
print sys.path
print sys.executable
import tensorflow as tf
import numpy as np
import vtk
import argparse
import os
import util
import layers as tf_util
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('vtkfolder')
parser.add_argument('modality', choices=['ct','mr'])

args = parser.parse_args()

folder = os.path.abspath(args.vtkfolder)
if not folder.endswith('/'):
    folder = folder+'/'
modality = args.modality

#################################
# Define Parameters
#################################
C=1
crop_dims = 128
Nbatch = 16
init = 6e-2
Nfilters = 32
EPS=1e-4
leaky_relu = tf.contrib.keras.layers.LeakyReLU(0.2)
ISOVALUE = 0.1
#################################
# Build tensorflow model
#################################
with tf.device('/cpu:0'):
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
saver.restore(sess,'/home/marsdenlab/projects/SV3/ml_plugin/org.sv.gui.qt.ml/src/python/models/i2i_CT/i2i_CT')

#################################
# Get vts files
#################################
print "Reading vtk structured points files"
vts_files = os.listdir(folder)
vts_files = [v for v in vts_files if '.vts' in v and not '_seg.vts' in v]
vts_files = [folder+v for v in vts_files]

vts_vtks = [util.readVTKSP(v) for v in vts_files]

vts_nps = [util.VTKSPtoNumpyFromFile(v) for v in vts_files]

if any([(v.shape[1]<crop_dims or v.shape[2] < crop_dims) for v in vts_nps]):
    raise RuntimeError("Error Vtk structured points files have dimension smaller than 128x128, need at least 128x128")

#Crop and normalize the images
vts_nps = [util.crop_center_nd(v,crop_dims,crop_dims) for v in vts_nps]
vts_nps = np.asarray(vts_nps)[:,0,:,:,np.newaxis].astype(float)
print vts_nps.shape

if modality == 'ct':
    vts_nps = 1.0*(vts_nps+1000)/2000
if modality == 'mr':
    raise RuntimeError('MR Not implemented yet')

#Need there to be a multiple of Nbatch images
print "Padding images to be multiple of Nbatch"
NUMBER_OF_IMAGES = vts_nps.shape[0]

r = vts_nps.shape[0] % Nbatch
if not r==0:
    r = Nbatch-r
    pad_images = np.zeros((r,crop_dims,crop_dims,1))
    vts_nps = np.concatenate((vts_nps,pad_images),axis=0)

if not vts_nps.shape[0]%Nbatch == 0:
    raise RuntimeError('Error adding images to be multiple of batch size')
print vts_nps.shape
###############################
#Segment
###############################
print "Segmenting images"
segmented_images = sess.run(yclass,{x:vts_nps})
segmented_images = segmented_images[:NUMBER_OF_IMAGES]

#TEST
# for i in range(NUMBER_OF_IMAGES):
#     index = vts_files[i].split('/')[-1].replace('.vts','')
#     plt.figure()
#     plt.imshow(segmented_images[i,:,:,0])
#     plt.colorbar()
#     plt.savefig(folder+'{}_seg.png'.format(index),dpi=300)
#     plt.close()
#
#     plt.figure()
#     plt.imshow(vts_nps[i,:,:,0])
#     plt.colorbar()
#     plt.savefig(folder+'{}_img.png'.format(index),dpi=300)
#     plt.close()
#convert segmented images to structured points
segmented_vts = []
for i in range(NUMBER_OF_IMAGES):
    v = util.VTKNumpytoSP(segmented_images[i,:,:,0])
    sp = vts_vtks[i]
    spacing = sp.GetSpacing()
    origin = [-64*spacing[0],-64*spacing[1],0.0]
    v.SetOrigin(origin)
    #v.SetOrigin([0,0,0])
    v.SetSpacing(spacing)
    util.writeSP(v,vts_files[i].replace('.vts','_seg.vts'))
    segmented_vts.append(v)

print "Extracting contours"
#contours = [util.marchingSquares(s[:,:,0],iso=ISOVALUE,mode='center') for s in segmented_images]
contours = [util.marchingSquares(s,iso=ISOVALUE,mode='center',asNumpy=False) for s in segmented_vts]
contours = [util.VTKPDPointstoNumpy(c)[:,:2] for c in contours]
print contours[0]
contours = [util.reorder_contour(c) for c in contours]
contours = [util.numpyToPd(c) for c in contours]

print "Writing polydatas"
for i in range(NUMBER_OF_IMAGES):
    fn = vts_files[i].replace('.vts','.vtp')
    pd = contours[i]
    util.writePolydata(pd,fn)
