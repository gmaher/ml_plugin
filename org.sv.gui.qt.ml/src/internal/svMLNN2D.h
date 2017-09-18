#ifndef SVMLNN2D_H
#define SVMLNN2D_H


#include "cvStrPts.h"
#include "svPathElement.h"

static const int SIZE = 128;
static const std::string TEMP_DIR = "tmp"
int writeResliceImage(svPathElement::svPathPoint pathPoint, vtkImageData* volumeimage, int type);

#endif // SVMLNN2D_H
