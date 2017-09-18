#ifndef SVMLCONTOURGROUPCREATE_H
#define SVMLCONTOURGROUPCREATE_H

#include "svDataNodeOperationInterface.h"
#include "svContourGroup.h"
#include <mitkDataStorage.h>
#include <QWidget>

namespace Ui {
class svMLContourGroupCreate;
}

class svMLContourGroupCreate : public QWidget
{
    Q_OBJECT

public:

    svMLContourGroupCreate(mitk::DataStorage::Pointer dataStorage, mitk::DataNode::Pointer selectedNode, int timeStep);

    virtual ~svMLContourGroupCreate();

//    void SetPreferencedValues(svLoftingParam* param);

public slots:

    void CreateGroup();

    void Cancel();

    void SetFocus();

    void Activated();

protected:

    Ui::svMLContourGroupCreate *ui;

    QWidget* m_Parent;

    mitk::DataNode::Pointer m_SegFolderNode;

    mitk::DataNode::Pointer m_PathFolderNode;

    mitk::DataStorage::Pointer m_DataStorage;

    mitk::DataNode::Pointer m_SelecteNode;

    int m_TimeStep;

    svDataNodeOperationInterface* m_Interface;
};

#endif // SVMLCONTOURGROUPCREATE_H
