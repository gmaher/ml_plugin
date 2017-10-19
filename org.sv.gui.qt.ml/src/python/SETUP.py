import os

#Tensorflow wheels
TF_LINUX = "https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl"
#TF_LINUX = "https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp35-cp35m-linux_x86_64.whl"
#TF_OSX   = "https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl"
TF_OSX   = "https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py2-none-any.whl"
TF_WIN   = "https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-0.12.1-cp35-cp35m-win_amd64.whl"

#Set up environment info
OS        = os.name
VTK_URL   = 'vtk'
HOME_DIR  = os.environ['HOME']
cwd       = os.path.abspath(os.getcwd())
ANACONDA  = "Anaconda2-4.2.0-Linux-x86_64.sh"
CONDA_DIR = cwd+'/anaconda'
PIP       = cwd+'/anaconda/bin/pip'
CONDA     = cwd+'/anaconda/bin/conda'
ENV_KEY   = 'SV_ML_HOME'

print "OS: {}, Installing anaconda 2, tensorflow and vtk python packages in {}".format(OS,CONDA_DIR)

if OS == 'posix':
    tf_url = TF_LINUX
if OS == 'mac':
    tf_url = TF_OSX
if OS == 'nt':
    tf_url = TF_WIN
    raise RuntimeError("Don't use windows please")

# print "Installing anaconda in directory: {}".format(cwd)
print "Installing anaconda\n"
r = os.system('{}/{} -b -p {}'.format(cwd,ANACONDA,CONDA_DIR))
if not r == 0: raise RuntimeError('Failed to install anaconda')

print "Installing Tensorflow\n"
r = os.system('{} install -I {}'.format(PIP,tf_url))
if not r == 0: raise RuntimeError('Failed to pip install tensorflow')

print "Installing VTK\n"
r = os.system('{} install -y {}'.format(CONDA,VTK_URL))
if not r == 0: raise RuntimeError('Failed to pip install vtk')

#Now set environment variables needed by plugin
#First check that it isnt already there
with open('{}/.bashrc'.format(HOME_DIR),'r') as f:
    lines = f.readlines()
    if any([ENV_KEY in l for l in lines]): raise RuntimeError("environment variables {}\
    already exists, delete it before installing sv ml plugin".format(ENV_KEY))

print "Adding environment variable {} to .bashrc file".format(ENV_KEY)
s = 'export {}={}'.format(ENV_KEY,cwd)
os.system("echo {} >> $HOME/.bashrc".format(s))
