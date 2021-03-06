#ifndef SVMLNN2D_H
#define SVMLNN2D_H

#include "cvStrPts.h"
#include "svPathElement.h"
#include "svMLNN2D.h"
#include "svSegmentationUtils.h"
#include "svContour.h"
#include "svContourGroup.h"
#include "cvPolyData.h"
#include "cv_sys_geom.h"
#include <mitkNodePredicateDataType.h>
#include <mitkDataStorage.h>
#include <QDir>
#include <QString>
#include "svDataNodeOperation.h"
#include <vtkStructuredPoints.h>
#include <vtkStructuredPointsWriter.h>
#include <vtkPolyDataReader.h>
#include <vtkSmartPointer.h>
#include <sstream>
#include <mitkIOUtil.h>

static const int SIZE = 128;
static const std::string TEMP_DIR_PATH = "tmp";
static const std::string STORE_PATH = "nn2d_seg";
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

  svContour* getContour(svPathElement::svPathPoint pathPoint, int posID){

    QDir dir = getDir();
    std::string folder = dir.absolutePath().toStdString();

    vtkSmartPointer<vtkPolyDataReader> reader = vtkPolyDataReader::New();

    std::stringstream ss;
    ss << folder;
    ss << "/";
    ss << TEMP_DIR_PATH;
    ss << "/";
    ss << posID;
    ss << ".vtp";
    std::string fn = ss.str();

    reader->SetFileName(fn.c_str());
    reader->Update();

    vtkPolyData* pd = reader->GetOutput();
    cvPolyData* cv_oriented = orientProfile(pathPoint,pd);
    vtkPolyData* pd_oriented = cv_oriented->GetVtkPolyData();

    //insert pd points into contour
    svContour* contour=new svContour();
    contour->SetPathPoint(pathPoint);
    contour->SetPlaced(true);
    contour->SetMethod("NN2D");

    std::vector<mitk::Point3D> contourPoints;

    bool ifClosed;
    std::deque<int> IDList=svSegmentationUtils::GetOrderedPtIDs(pd_oriented->GetLines(),ifClosed);
    double point[3];
    mitk::Point3D pt;
    for(int i=0;i<IDList.size();i++)
    {
        pd_oriented->GetPoint(IDList[i],point);
        pt[0]=point[0];
        pt[1]=point[1];
        pt[2]=point[2];
        contourPoints.push_back(pt);
    }

    contour->SetClosed(ifClosed);
    std::cout << "set contour";
    contour->SetContourPoints(contourPoints);

    return contour;
  }

  cvPolyData* orientProfile(svPathElement::svPathPoint pathPoint, vtkPolyData* contour){
    cvPolyData *dst = new cvPolyData(contour);
    cvPolyData* dst_merged;
    double tol=0.001;
    dst_merged=sys_geom_MergePts_tol(dst, tol );

    double pos[3],nrm[3],xhat[3];

    pos[0]=pathPoint.pos[0];
    pos[1]=pathPoint.pos[1];
    pos[2]=pathPoint.pos[2];

    nrm[0]=pathPoint.tangent[0];
    nrm[1]=pathPoint.tangent[1];
    nrm[2]=pathPoint.tangent[2];

    xhat[0]=pathPoint.rotation[0];
    xhat[1]=pathPoint.rotation[1];
    xhat[2]=pathPoint.rotation[2];

    cvPolyData *dst2;

    sys_geom_OrientProfile(dst_merged, pos, nrm,xhat,&dst2);

    return dst2;
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
    QString Qstore_dir = QString::QString(STORE_PATH.c_str());

    m_TmpDir = Qtmp_dir;
    m_StoreDir = Qstore_dir;

    if(!dir.exists(Qtmp_dir))
    {
      std::cout <<"directory doesnt exist, creating\n";
        dir.mkdir(Qtmp_dir);
    }
    else {
      std::cout << "directory exists removing\n";
      dir.cd(Qtmp_dir);
      dir.removeRecursively();
      dir = getDir();
      dir.mkdir(Qtmp_dir);
    }

    if(!dir.exists(Qstore_dir))
    {
      std::cout <<"directory doesnt exist, creating\n";
        dir.mkdir(Qstore_dir);
    }

  }

  int computeSegmentations(std::string modality){


    const char* sv_ml_home = std::getenv("SV_ML_HOME");
    if (!sv_ml_home){
      std::cout <<"SV_ML_HOME environment variable missing\n";
    }

    std::stringstream py_home;
    py_home << sv_ml_home;
    py_home << "/anaconda";

    std::stringstream python;
    python << py_home.str();
    python << "/bin/python";

    std::stringstream script;
    script << sv_ml_home;
    script << "/segment2d.py";

    QDir dir = getDir();

    std::stringstream ss;
    ss << "export PYTHONHOME=" << py_home.str() << " && ";
    ss << python.str() << " ";
    ss << script.str() << " ";
    ss << dir.absolutePath().toStdString() << "/";
    ss << TEMP_DIR_PATH << " ";
    ss << modality;
    std::cout << ss.str() << "\n";
    return system(ss.str().c_str());
  }

  void segment(mitk::DataStorage* dataStorage){
    std::cout << "NN segmenting\n";

    makeDir(dataStorage);

  }

  void saveSegmentations(std::string groupName){


    mitk::NodePredicateDataType::Pointer TypeCondition = mitk::NodePredicateDataType::New("svContourGroup");
    mitk::DataStorage::SetOfObjects::ConstPointer rs=m_DS->GetSubset(TypeCondition);

    QDir dir = getDir();
    dir.cd(m_StoreDir);

    for (int i =0; i < rs->size(); i++){
      mitk::DataNode::Pointer node=rs->GetElement(i);
      svContourGroup *contourGroup=dynamic_cast<svContourGroup*>(node->GetData());

      if(contourGroup && node->GetName()==groupName)
      {

          QString	filePath=dir.absoluteFilePath(QString::fromStdString(node->GetName())+".ctgr");
          mitk::IOUtil::Save(node->GetData(),filePath.toStdString());
          node->SetStringProperty("path",dir.absolutePath().toStdString().c_str());
          contourGroup->SetDataModified(false);
          std::cout << "Saved contour group " << groupName << "\n";
      }
    }

  }


private:

  mitk::DataStorage* m_DS;
  QString m_TmpDir;
  QString m_StoreDir;
};

#endif // SVMLNN2D_H
