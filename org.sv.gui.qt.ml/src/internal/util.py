import numpy as np
import vtk

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

def marchingSquares(img, iso=0.0, mode='center'):
    s = img.shape
    alg = vtk.vtkMarchingSquares()

    sp = VTKNumpytoSP(img)

    alg.SetInputData(sp)
    alg.SetValue(0,iso)
    alg.Update()
    pds = alg.GetOutput()

    a = vtk.vtkPolyDataConnectivityFilter()
    a.SetInputData(pds)

    if mode=='center':
        a.SetExtractionModeToClosestPointRegion()
        a.SetClosestPoint(float(s[0])/2,float(s[1])/2,0.0)

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
