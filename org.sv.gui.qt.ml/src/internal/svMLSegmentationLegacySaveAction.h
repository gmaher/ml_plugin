#ifndef SVMLSEGMENTATIONLEGACYSAVEACTION_H
#define SVMLSEGMENTATIONLEGACYSAVEACTION_H

#include <org_sv_gui_qt_segmentation_Export.h>

#include <mitkIContextMenuAction.h>
#include <mitkDataNode.h>

#include <QObject>

class SV_QT_SEGMENTATION svMLSegmentationLegacySaveAction : public QObject, public mitk::IContextMenuAction
{
  Q_OBJECT
  Q_INTERFACES(mitk::IContextMenuAction)

public:
  svMLSegmentationLegacySaveAction();
  ~svMLSegmentationLegacySaveAction();

  // IContextMenuAction
  void Run(const QList<mitk::DataNode::Pointer> &selectedNodes) override;
  void SetDataStorage(mitk::DataStorage *dataStorage) override;
  void SetSmoothed(bool smoothed) override {}
  void SetDecimated(bool decimated) override {}
  void SetFunctionality(berry::QtViewPart *functionality) override {}

private:
  svMLSegmentationLegacySaveAction(const svMLSegmentationLegacySaveAction &);
  svMLSegmentationLegacySaveAction & operator=(const svMLSegmentationLegacySaveAction &);

  mitk::DataStorage::Pointer m_DataStorage;

};

#endif
