import sys
#sys.path += ['', '/home/marsdenlab/anaconda2/lib/python2.7/site-packages/google/','/home/marsdenlab/anaconda2/lib/python2.7/site-packages/google/protobuf','/home/marsdenlab/libraries/vmtk/build/Install/lib/python2.7/site-packages', '/home/marsdenlab/projects/SV3', '/home/marsdenlab/anaconda2/lib/python27.zip', '/home/marsdenlab/anaconda2/lib/python2.7', '/home/marsdenlab/anaconda2/lib/python2.7/plat-linux2', '/home/marsdenlab/anaconda2/lib/python2.7/lib-tk', '/home/marsdenlab/anaconda2/lib/python2.7/lib-old', '/home/marsdenlab/anaconda2/lib/python2.7/lib-dynload', '/home/marsdenlab/anaconda2/lib/python2.7/site-packages', '/home/marsdenlab/anaconda2/lib/python2.7/site-packages/Sphinx-1.5.4-py2.7.egg', '/home/marsdenlab/libraries/pybedtools', '/home/marsdenlab/projects/tcl_code/python/src/pyemd', '/home/marsdenlab/anaconda2/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg']
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
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
#from skimage.segmentation import chan_vese
parser = argparse.ArgumentParser()
parser.add_argument('vtkfolder')
#parser.add_argument('modality', choices=['ct','mr','coronary'])

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
ISO_SEG = 0.35
ISOVALUE = 0.5
lam = 1e-3
lr = 1e-4
ERODE_ITERS = 2
SMOOTH_FACTOR = 0.95
MAX_LS_ITER = 10
#################################
# Build tensorflow model
#################################
with tf.device('/cpu:0'):

    #################################
    # Load model
    #################################
    from dl_template.experiment.I2INet import Model
    from dl_template.modules import io

    global_config_file = os.environ['SV_ML_HOME']+'/config/global.yaml'
    case_config_file = os.environ['SV_ML_HOME']+'/config/case1_perturb15.yaml'

    global_config = io.load_yaml(global_config_file)
    case_config   = io.load_yaml(case_config_file)

    model = Model(global_config, case_config)
    model.load()

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
M = len(vts_nps)
V = np.zeros((M,crop_dims,crop_dims,1))
max_ = np.amax(vts_nps)
min_ = np.amin(vts_nps)
for i in range(M):
    v = vts_nps[i][0]
    print v.shape
    V[i] = util.crop_center(v,crop_dims,crop_dims).reshape(128,128,1)
    V[i] = (1.0*V[i]-np.mean(V[i]))/(np.std(V[i]+1e-5))

vts_nps = V

print vts_nps.shape

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
#TEST
for i in range(NUMBER_OF_IMAGES):
    index = vts_files[i].split('/')[-1].replace('.vts','')

    plt.figure()
    plt.imshow(vts_nps[i,:,:,0],cmap='gray')
    plt.colorbar()
    plt.savefig(folder+'{}_img_before.png'.format(index),dpi=300)
    plt.close()

print "Segmenting images"
segmented_images = []
for i in range(0,vts_nps.shape[0],Nbatch):
    segs = model.predict(vts_nps[i:i+Nbatch])

    if segmented_images == []:
        segmented_images = segs
    else:
        segmented_images = np.concatenate((segmented_images,segs))

segmented_images = segmented_images[:NUMBER_OF_IMAGES]

#TEST
for i in range(NUMBER_OF_IMAGES):
    index = vts_files[i].split('/')[-1].replace('.vts','')
    plt.figure()
    plt.imshow(segmented_images[i,:,:,0],cmap='gray')
    plt.colorbar()
    plt.savefig(folder+'{}_seg.png'.format(index),dpi=300)
    plt.close()

    plt.figure()
    plt.imshow(vts_nps[i,:,:,0],cmap='gray')
    plt.colorbar()
    plt.savefig(folder+'{}_img.png'.format(index),dpi=300)
    plt.close()

#THRESHOLD
segmented_images = util.threshold(segmented_images,ISO_SEG)

for i in range(NUMBER_OF_IMAGES):
    segmented_images[i,:,:,0] = binary_fill_holes(segmented_images[i,:,:,0])

#convert segmented images to structured points
segmented_vts = []
for i in range(NUMBER_OF_IMAGES):
    s = segmented_images[i,:,:,0]
    if np.sum(s)==0: continue
        #raise RuntimeError('empty segmentation produced at {}th image'.format(i))
    v = util.VTKNumpytoSP(s)
    sp = vts_vtks[i]
    spacing = sp.GetSpacing()
    origin = [-64*spacing[0],-64*spacing[1],0.0]
    v.SetOrigin(origin)
    #v.SetOrigin([0,0,0])
    v.SetSpacing(spacing)
    util.writeSP(v,vts_files[i].replace('.vts','_seg.vts'))
    segmented_vts.append(v)

print "Extracting contours"

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
