#ifndef SVMLSEGMENTATIONLEGACYLOADACTION_H
#define SVMLSEGMENTATIONLEGACYLOADACTION_H

#include <org_sv_gui_qt_segmentation_Export.h>

#include <svPath.h>

#include <mitkIContextMenuAction.h>
#include <mitkDataNode.h>

#include <QObject>

class SV_QT_SEGMENTATION svMLSegmentationLegacyLoadAction : public QObject, public mitk::IContextMenuAction
{
  Q_OBJECT
  Q_INTERFACES(mitk::IContextMenuAction)

public:
  svMLSegmentationLegacyLoadAction();
  ~svMLSegmentationLegacyLoadAction();

  // IContextMenuAction
  void Run(const QList<mitk::DataNode::Pointer> &selectedNodes) override;
  void SetDataStorage(mitk::DataStorage *dataStorage) override;
  void SetSmoothed(bool smoothed) override {}
  void SetDecimated(bool decimated) override {}
  void SetFunctionality(berry::QtViewPart *functionality) override {}

  svPath* GetPath(int groupPathID, std::string groupPathName, mitk::DataNode::Pointer segFolderNode);

private:
  svMLSegmentationLegacyLoadAction(const svMLSegmentationLegacyLoadAction &);
  svMLSegmentationLegacyLoadAction & operator=(const svMLSegmentationLegacyLoadAction &);

  mitk::DataStorage::Pointer m_DataStorage;

};

#endif
