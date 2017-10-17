import numpy as np
import vtk
from vtk import vtkImageExport
from vtk.util import numpy_support

def numpyToPd(d):
    '''
    input is a list of points
    '''
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()

    for p in d:
        x1 = p[0]
        x2 = p[1]
        x3 = 0
        pts.InsertNextPoint([x1,x2,x3])

    pd.SetPoints(pts)

    lines = vtk.vtkCellArray()
    for i in range(len(d)-1):
        l = vtk.vtkLine()
        l.GetPointIds().SetId(0,i)
        l.GetPointIds().SetId(1,i+1)
        lines.InsertNextCell(l)

    l = vtk.vtkLine()
    l.GetPointIds().SetId(0,len(d)-1)
    l.GetPointIds().SetId(1,0)
    lines.InsertNextCell(l)

    pd.SetLines(lines)

    return pd

def readVTKSP(fn):
	'''
	reads a vtk structured points object from a file
	'''
	sp_reader = vtk.vtkStructuredPointsReader()
	sp_reader.SetFileName(fn)
	sp_reader.Update()
	sp = sp_reader.GetOutput()
	return sp

def writeSP(sp,fn):
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetInputData(sp)
    writer.SetFileName(fn)
    writer.Write()

def writePolydata(pd,fn):
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(pd)
    writer.SetFileName(fn)
    writer.Write()

def crop_center_nd(img,cropx,cropy):
    s = img.shape
    startx = s[1]//2-(cropx//2)
    starty = s[2]//2-(cropy//2)
    return img[:,starty:starty+cropy,startx:startx+cropx]

def VTKSPtoNumpy(vol):
    '''
    Utility function to convert a VTK structured points (SP) object to a numpy array
    the exporting is done via the vtkImageExport object which copies the data
    from the supplied SP object into an empty pointer or array
    C/C++ can interpret a python string as a pointer/array
    This function was shamelessly copied from
    http://public.kitware.com/pipermail/vtkusers/2002-September/013412.html
    args:
    	@a vol: vtk.vtkStructuredPoints object
    '''
    exporter = vtkImageExport()
    exporter.SetInputData(vol)
    dims = exporter.GetDataDimensions()
    if np.sum(dims) == 0:
        raise RuntimeError('Error converting vtk structured points file to numpy array')
        return np.zeros((1,64,64))
    if (exporter.GetDataScalarType() == 3):
    	dtype = UnsignedInt8
    if (exporter.GetDataScalarType() == 4):
    	dtype = np.short
    if (exporter.GetDataScalarType() == 5):
    	dtype = np.int16
    if (exporter.GetDataScalarType() == 10):
    	dtype = np.float32
    if (exporter.GetDataScalarType() == 11):
    	dtype = np.float64
    a = np.zeros(reduce(np.multiply,dims),dtype)
    s = a.tostring()
    exporter.SetExportVoidPointer(s)
    exporter.Export()
    a = np.reshape(np.fromstring(s,dtype),(dims[2],dims[0],dims[1]))
    return a

def VTKSPtoNumpyFromFile(fn):
    '''
    reads a .vts file into a numpy array
    args:
    	@a fn - string, filename of .sp file to read
    '''
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(fn)
    reader.Update()
    sp = reader.GetOutput()

    if sp.GetNumberOfPoints() == 0:
        raise RuntimeError("Error reading vtk structure points file {}".format(fn))
    return VTKSPtoNumpy(sp)

def VTKNumpytoSP(img_):
    img = img_.T

    H,W = img.shape

    sp = vtk.vtkStructuredPoints()
    sp.SetDimensions(H,W,1)
    sp.AllocateScalars(10,1)
    for i in range(H):
        for j in range(W):
            v = img[i,j]
            sp.SetScalarComponentFromFloat(i,j,0,0,v)

    return sp

def VTKPDPointstoNumpy(pd):
	'''
	function to convert the points data of a vtk polydata object to a numpy array

	args:
		@a pd: vtk.vtkPolyData object
	'''
	return numpy_support.vtk_to_numpy(pd.GetPoints().GetData())


def marchingSquares(img, iso=0.0, mode='center',asNumpy=True):
    alg = vtk.vtkMarchingSquares()

    if asNumpy:
        sp = VTKNumpytoSP(img)
    else:
        sp = img
    alg.SetInputData(sp)
    alg.SetValue(0,iso)
    alg.Update()
    pds = alg.GetOutput()

    a = vtk.vtkPolyDataConnectivityFilter()
    a.SetInputData(pds)

    if mode=='center':
        a.SetExtractionModeToClosestPointRegion()
        a.SetClosestPoint(0.0,0.0,0.0)

    elif mode=='all':
        a.SetExtractionModeToAllRegions()

    a.Update()
    pds = a.GetOutput()

    return pds

def reorder_contour(c):
    N = len(c)

    even_inds = np.arange(0,N,2)
    odd_inds = np.arange(1,N,2)

    even_points = np.asarray([c[i] for i in even_inds])
    odd_points = np.asarray([c[i] for i in odd_inds])

    N_even = len(even_points)
    ret = np.zeros_like(c)
    ret[:N_even] = even_points
    ret[N_even:] = np.flipud(odd_points)
    ret = ret[:-2]
    return ret.copy()
