#ifndef SVMLCONTOURGROUPCREATEACTION_H
#define SVMLCONTOURGROUPCREATEACTION_H

#include <org_sv_gui_qt_segmentation_Export.h>

#include "svMLContourGroupCreate.h"

#include <mitkIContextMenuAction.h>
#include <mitkDataNode.h>

#include <QObject>

class SV_QT_SEGMENTATION svMLContourGroupCreateAction : public QObject, public mitk::IContextMenuAction
{
  Q_OBJECT
  Q_INTERFACES(mitk::IContextMenuAction)

public:
  svMLContourGroupCreateAction();
  ~svMLContourGroupCreateAction();

  // IContextMenuAction
  void Run(const QList<mitk::DataNode::Pointer> &selectedNodes) override;
  void SetDataStorage(mitk::DataStorage *dataStorage) override;
  void SetSmoothed(bool smoothed) override {}
  void SetDecimated(bool decimated) override {}
  void SetFunctionality(berry::QtViewPart *functionality) override;

private:
  svMLContourGroupCreateAction(const svMLContourGroupCreateAction &);
  svMLContourGroupCreateAction & operator=(const svMLContourGroupCreateAction &);

  mitk::DataStorage::Pointer m_DataStorage;
  berry::QtViewPart *m_Functionality;

  svMLContourGroupCreate* m_ContourGroupCreateWidget;
};

#endif
