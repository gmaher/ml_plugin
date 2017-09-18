#ifndef SVMLSEG3DCREATEACTION_H
#define SVMLSEG3DCREATEACTION_H

#include <org_sv_gui_qt_segmentation_Export.h>

#include "svDataNodeOperationInterface.h"

#include <mitkIContextMenuAction.h>
#include <mitkDataNode.h>

#include <QObject>

class SV_QT_SEGMENTATION svMLSeg3DCreateAction : public QObject, public mitk::IContextMenuAction
{
  Q_OBJECT
  Q_INTERFACES(mitk::IContextMenuAction)

public:
  svMLSeg3DCreateAction();
  ~svMLSeg3DCreateAction();

  // IContextMenuAction
  void Run(const QList<mitk::DataNode::Pointer> &selectedNodes) override;
  void SetDataStorage(mitk::DataStorage *dataStorage) override;
  void SetSmoothed(bool smoothed) override {}
  void SetDecimated(bool decimated) override {}
  void SetFunctionality(berry::QtViewPart *functionality) override;

private:
  svMLSeg3DCreateAction(const svMLSeg3DCreateAction &);
  svMLSeg3DCreateAction & operator=(const svMLSeg3DCreateAction &);

  mitk::DataStorage::Pointer m_DataStorage;
  berry::QtViewPart *m_Functionality;

  svDataNodeOperationInterface* m_Interface;
};

#endif
