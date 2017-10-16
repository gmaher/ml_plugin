#include "svMLNN2D.h"
#include "svSegmentationUtils.h"
#include <mitkNodePredicateDataType.h>
#include <mitkDataStorage.h>
#include <QDir>
#include <QString>
#include "svDataNodeOperation.h"


svMLNN2D::svMLNN2D(){

}

svMLNN2D::~svMLNN2D(){

}
int svMLNN2D::writeResliceImage(svPathElement::svPathPoint pathPoint, vtkImageData* volumeimage, int type){

  cvStrPts* strPts=svSegmentationUtils::GetSlicevtkImage(pathPoint, volumeimage,  SIZE);
}

void svMLNN2D::makeDir(){
  mitk::NodePredicateDataType::Pointer isProjFolder = mitk::NodePredicateDataType::New("svProjectFolder");
  mitk::DataNode::Pointer projFolderNode=GetDataStorage()->GetNode (isProjFolder);

  std::string projPath="";
  projFolderNode->GetStringProperty("project path", projPath);

  QString QprojPath = QString::QString(projPath.c_str());
  QString Qtmp_dir = QString::QString(TEMP_DIR_PATH.c_str());

  QDir dir(QprojPath);

  if(!dir.exists(Qtmp_dir))
  {
      dir.mkdir(Qtmp_dir);
  }

  dir.cd(Qtmp_dir);
}

void svMLNN2D::segment(){
  std::cout << "NN segmenting\n";

  makeDir();
}
