#ifndef SVMLCONTOURGROUPPOINT2DSIZEACTION_H
#define SVMLCONTOURGROUPPOINT2DSIZEACTION_H

#include <org_sv_gui_qt_segmentation_Export.h>

#include <mitkIContextMenuAction.h>
#include <mitkDataNode.h>

#include <QObject>

class SV_QT_SEGMENTATION svMLContourGroupPoint2DSizeAction : public QObject, public mitk::IContextMenuAction
{
  Q_OBJECT
  Q_INTERFACES(mitk::IContextMenuAction)

public:
  svMLContourGroupPoint2DSizeAction();
  ~svMLContourGroupPoint2DSizeAction();

  // IContextMenuAction
  void Run(const QList<mitk::DataNode::Pointer> &selectedNodes) override;
  void SetDataStorage(mitk::DataStorage *dataStorage) override;
  void SetSmoothed(bool smoothed) override {}
  void SetDecimated(bool decimated) override {}
  void SetFunctionality(berry::QtViewPart *functionality) override;

private:
  svMLContourGroupPoint2DSizeAction(const svMLContourGroupPoint2DSizeAction &);
  svMLContourGroupPoint2DSizeAction & operator=(const svMLContourGroupPoint2DSizeAction &);

  mitk::DataStorage::Pointer m_DataStorage;
  berry::QtViewPart *m_Functionality;

};

#endif
