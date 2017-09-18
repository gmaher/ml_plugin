#ifndef SVMLCONTOURGROUPPOINT3DSIZEACTION_H
#define SVCONTOURGROUPHPOINT3DSIZEACTION_H

#include <org_sv_gui_qt_segmentation_Export.h>

#include <mitkIContextMenuAction.h>
#include <mitkDataNode.h>

#include <QObject>

class SV_QT_SEGMENTATION svMLContourGroupPoint3DSizeAction : public QObject, public mitk::IContextMenuAction
{
  Q_OBJECT
  Q_INTERFACES(mitk::IContextMenuAction)

public:
  svMLContourGroupPoint3DSizeAction();
  ~svMLContourGroupPoint3DSizeAction();

  // IContextMenuAction
  void Run(const QList<mitk::DataNode::Pointer> &selectedNodes) override;
  void SetDataStorage(mitk::DataStorage *dataStorage) override;
  void SetSmoothed(bool smoothed) override {}
  void SetDecimated(bool decimated) override {}
  void SetFunctionality(berry::QtViewPart *functionality) override;

private:
  svMLContourGroupPoint3DSizeAction(const svMLContourGroupPoint3DSizeAction &);
  svMLContourGroupPoint3DSizeAction & operator=(const svMLContourGroupPoint3DSizeAction &);

  mitk::DataStorage::Pointer m_DataStorage;
  berry::QtViewPart *m_Functionality;

};

#endif
