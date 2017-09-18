#include "svMLNN2D.h"
#include "svSegmentationUtils.h"
#include <mitkNodePredicateDataType.h>
#include <mitkDataStorage.h>

svMLNN2D::writeResliceImage(svPathElement::svPathPoint pathPoint, vtkImageData* volumeimage, int type){

  mitk::NodePredicateDataType::Pointer isProjFolder = mitk::NodePredicateDataType::New("svProjectFolder");
  mitk::DataNode::Pointer projFolderNode=GetDataStorage()->GetNode (isProjFolder);

  std::string projPath="";
  projFolderNode->GetStringProperty("project path", projPath);

  QDir dir(TEMP_DIR);

  if(!dir.exists())
  {
      dir.mkdir(TEMP_DIR);
  }

  dir.cd(TEMP_DIR);

  cvStrPts* strPts=GetSlicevtkImage(pathPoint, volumeimage,  SIZE);
}
