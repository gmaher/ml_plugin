#include "svMLContourGroupCreate.h"
#include "ui_svMLContourGroupCreate.h"

#include "svPath.h"
#include "svDataNodeOperation.h"
#include "svMLLoftingUtils.h"

#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <mitkNodePredicateDataType.h>
#include <mitkUndoController.h>
#include <mitkOperationEvent.h>

#include <QMessageBox>
#include <QFileDialog>

#include <iostream>
using namespace std;

svMLContourGroupCreate::svMLContourGroupCreate(mitk::DataStorage::Pointer dataStorage, mitk::DataNode::Pointer selectedNode, int timeStep)
    : ui(new Ui::svMLContourGroupCreate)
    , m_DataStorage(dataStorage)
    , m_SelecteNode(selectedNode)
    , m_TimeStep(timeStep)
    , m_SegFolderNode(NULL)
    , m_PathFolderNode(NULL)
{
    m_Interface=new svDataNodeOperationInterface;

    ui->setupUi(this);
    connect(ui->buttonBox, SIGNAL(accepted()), this, SLOT(CreateGroup()));
    connect(ui->buttonBox, SIGNAL(rejected()), this, SLOT(Cancel()));
    connect(ui->lineEditGroupName, SIGNAL(returnPressed()), this, SLOT(CreateGroup()));
    move(400,400);

    Activated();
}

svMLContourGroupCreate::~svMLContourGroupCreate()
{
    delete ui;
}

void svMLContourGroupCreate::Activated()
{
    ui->comboBox->clear();

    m_PathFolderNode=NULL;
    m_SegFolderNode=NULL;

    if(m_SelecteNode.IsNull())
        return;

    mitk::DataNode::Pointer selectedNode=m_SelecteNode;

    mitk::NodePredicateDataType::Pointer isProjFolder = mitk::NodePredicateDataType::New("svProjectFolder");
    mitk::DataStorage::SetOfObjects::ConstPointer rs=m_DataStorage->GetSources (selectedNode,isProjFolder,false);

    if(rs->size()>0)
    {
        mitk::DataNode::Pointer projFolderNode=rs->GetElement(0);

        mitk::NodePredicateDataType::Pointer isSegFolder = mitk::NodePredicateDataType::New("svSegmentationFolder");
        mitk::NodePredicateDataType::Pointer isGroupNode = mitk::NodePredicateDataType::New("svContourGroup");

        if(isSegFolder->CheckNode(selectedNode)){
            m_SegFolderNode=selectedNode;
        }else if(isGroupNode->CheckNode(selectedNode)){
            mitk::DataStorage::SetOfObjects::ConstPointer rs = m_DataStorage->GetSources(selectedNode);
            if(rs->size()>0){
                m_SegFolderNode=rs->GetElement(0);
            }
        }

        rs=m_DataStorage->GetDerivations(projFolderNode,mitk::NodePredicateDataType::New("svPathFolder"));
        if (rs->size()>0)
        {
            m_PathFolderNode=rs->GetElement(0);

            rs=m_DataStorage->GetDerivations(m_PathFolderNode,mitk::NodePredicateDataType::New("svPath"));

            for(int i=0;i<rs->size();i++)
            {
                ui->comboBox->addItem(QString::fromStdString(rs->GetElement(i)->GetName()));
            }
        }

    }

    ui->lineEditGroupName->clear();
}

void svMLContourGroupCreate::SetFocus( )
{
    ui->comboBox->setFocus();
}

void svMLContourGroupCreate::CreateGroup()
{
    QString selectedPathName=ui->comboBox->currentText();
    if(selectedPathName=="")
    {
        QMessageBox::warning(NULL,"No Path Selected","Please select a path!");
        return;
    }

    mitk::DataNode::Pointer selectedPathNode=m_DataStorage->GetNamedDerivedNode(selectedPathName.toStdString().c_str(),m_PathFolderNode);

    if(selectedPathNode.IsNull())
    {
        QMessageBox::warning(NULL,"No Path Found!","Please select a existing path!");
        return;
    }

    std::string groupName=ui->lineEditGroupName->text().trimmed().toStdString();

    if(groupName==""){
        groupName=selectedPathNode->GetName();
    }

    mitk::DataNode::Pointer exitingNode=m_DataStorage->GetNamedDerivedNode(groupName.c_str(),m_SegFolderNode);
    if(exitingNode){
        QMessageBox::warning(NULL,"Contour Group Already Created","Please use a different group name!");
        return;
    }

    svContourGroup::Pointer group = svContourGroup::New();

    svMLLoftingUtils::SetPreferencedValues(group->GetLoftingParam());

    group->SetPathName(selectedPathNode->GetName());
    group->SetDataModified();

    svPath* selectedPath=dynamic_cast<svPath*>(selectedPathNode->GetData());
    if(selectedPath)
    {
        group->SetPathID(selectedPath->GetPathID());
    }

    mitk::DataNode::Pointer groupNode = mitk::DataNode::New();
    groupNode->SetData(group);
    groupNode->SetName(groupName);

    float point2DSize=0;
    float pointSize=0;
    float resliceSize=0;
    if(m_SelecteNode.IsNotNull())
    {
        m_SelecteNode->GetFloatProperty("point.displaysize",point2DSize);
        m_SelecteNode->GetFloatProperty("point.3dsize",pointSize);
        m_SelecteNode->GetFloatProperty("reslice size",resliceSize);
        if(resliceSize==0)
        {
            svContourGroup* originalGroup=dynamic_cast<svContourGroup*>(m_SelecteNode->GetData());
            if(originalGroup)
                resliceSize=originalGroup->GetResliceSize();
        }
    }

    if(point2DSize!=0)
    {
        groupNode->SetFloatProperty("point.displaysize",point2DSize);
        group->SetProp("point 2D display size",QString::number(point2DSize).toStdString());
    }
    if(pointSize!=0)
    {
        groupNode->SetFloatProperty("point.3dsize",pointSize);
        group->SetProp("point size",QString::number(pointSize).toStdString());
    }
    if(resliceSize!=0)
        group->SetResliceSize(resliceSize);

//    m_DataStorage->Add(groupNode,m_SegFolderNode);
    mitk::OperationEvent::IncCurrObjectEventId();

    bool undoEnabled=true;
    svDataNodeOperation* doOp = new svDataNodeOperation(svDataNodeOperation::OpADDDATANODE,m_DataStorage,groupNode,m_SegFolderNode);
    if(undoEnabled)
    {
        svDataNodeOperation* undoOp = new svDataNodeOperation(svDataNodeOperation::OpREMOVEDATANODE,m_DataStorage,groupNode,m_SegFolderNode);
        mitk::OperationEvent *operationEvent = new mitk::OperationEvent(m_Interface, doOp, undoOp, "Add DataNode");
        mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
    }
    m_Interface->ExecuteOperation(doOp);

    hide();
}

void svMLContourGroupCreate::Cancel()
{
    hide();
}

