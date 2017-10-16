#ifndef SVMLNN2D_H
#define SVMLNN2D_H

#include "cvStrPts.h"
#include "svPathElement.h"
#include "svMLNN2D.h"
#include "svSegmentationUtils.h"
#include <mitkNodePredicateDataType.h>
#include <mitkDataStorage.h>
#include <QDir>
#include <QString>
#include "svDataNodeOperation.h"
#include <vtkStructuredPoints.h>
#include <vtkStructuredPointsWriter.h>
#include <vtkSmartPointer.h>
#include <sstream>

static const int SIZE = 128;
static const std::string TEMP_DIR_PATH = "tmp";

class svMLNN2D{

public:
  int writeResliceImage(svPathElement::svPathPoint pathPoint, vtkImageData* volumeimage, int posID){

    QDir dir = getDir();
    std::string folder = dir.absolutePath().toStdString();

    vtkSmartPointer<vtkStructuredPointsWriter> writer = vtkStructuredPointsWriter::New();

    cvStrPts* strPts=svSegmentationUtils::GetSlicevtkImage(pathPoint, volumeimage,  SIZE);

    vtkStructuredPoints* Pts = strPts->GetVtkStructuredPoints();

    writer->SetInputData(Pts);

    std::stringstream ss;
    ss << folder;
    ss << "/";
    ss << TEMP_DIR_PATH;
    ss << "/";
    ss << posID;
    ss << ".vts";
    std::string fn = ss.str();

    writer->SetFileName(fn.c_str());
    writer->Write();
  }

  QDir getDir(){
    mitk::NodePredicateDataType::Pointer isProjFolder = mitk::NodePredicateDataType::New("svProjectFolder");
    mitk::DataNode::Pointer projFolderNode=m_DS->GetNode (isProjFolder);

    std::string projPath="";
    projFolderNode->GetStringProperty("project path", projPath);

    QString QprojPath = QString::QString(projPath.c_str());

    QDir dir(QprojPath);

    return dir;
  }

  void makeDir(mitk::DataStorage* dataStorage){
    m_DS = dataStorage;

    QDir dir = getDir();
    QString Qtmp_dir = QString::QString(TEMP_DIR_PATH.c_str());
    m_TmpDir = Qtmp_dir;

    if(!dir.exists(Qtmp_dir))
    {
        dir.mkdir(Qtmp_dir);
    }

  }

  void segment(mitk::DataStorage* dataStorage){
    std::cout << "NN segmenting\n";

    makeDir(dataStorage);
  }

private:

  mitk::DataStorage* m_DS;
  QString m_TmpDir;
};

#endif // SVMLNN2D_H
