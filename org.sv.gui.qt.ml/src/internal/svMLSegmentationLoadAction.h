#ifndef SVMLSEGMENTATIONLOADACTION_H
#define SVMLSEGMENTATIONLOADACTION_H

#include <org_sv_gui_qt_segmentation_Export.h>

#include <svPath.h>

#include <mitkIContextMenuAction.h>
#include <mitkDataNode.h>

#include <QObject>

class SV_QT_SEGMENTATION svMLSegmentationLoadAction : public QObject, public mitk::IContextMenuAction
{
  Q_OBJECT
  Q_INTERFACES(mitk::IContextMenuAction)

public:
  svMLSegmentationLoadAction();
  ~svMLSegmentationLoadAction();

  // IContextMenuAction
  void Run(const QList<mitk::DataNode::Pointer> &selectedNodes) override;
  void SetDataStorage(mitk::DataStorage *dataStorage) override;
  void SetSmoothed(bool smoothed) override {}
  void SetDecimated(bool decimated) override {}
  void SetFunctionality(berry::QtViewPart *functionality) override {}

private:
  svMLSegmentationLoadAction(const svMLSegmentationLoadAction &);
  svMLSegmentationLoadAction & operator=(const svMLSegmentationLoadAction &);

  mitk::DataStorage::Pointer m_DataStorage;

};

#endif
