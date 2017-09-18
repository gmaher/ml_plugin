#include "svMLContourGroupPoint2DSizeAction.h"

#include "svContourGroup.h"

#include <mitkNodePredicateDataType.h>

#include <QInputDialog>

svMLContourGroupPoint2DSizeAction::svMLContourGroupPoint2DSizeAction()
    : m_Functionality(NULL)
{
}

svMLContourGroupPoint2DSizeAction::~svMLContourGroupPoint2DSizeAction()
{
}

void svMLContourGroupPoint2DSizeAction::Run(const QList<mitk::DataNode::Pointer> &selectedNodes)
{
    mitk::DataNode::Pointer selectedNode = selectedNodes[0];

    mitk::NodePredicateDataType::Pointer isGroup = mitk::NodePredicateDataType::New("svContourGroup");

    float initialValue=8.0f;
    if(isGroup->CheckNode(selectedNode))
        selectedNode->GetFloatProperty ("point.displaysize", initialValue);

    bool ok;
    float size=QInputDialog::getDouble(NULL, "Point 2D Display Size", "Size:", initialValue, 0.1, 100.0, 1, &ok);

    if(!ok)
        return;

    mitk::DataNode::Pointer segFolderNode=NULL;
    if(selectedNode.IsNotNull())
    {
        mitk::DataStorage::SetOfObjects::ConstPointer rs = m_DataStorage->GetSources(selectedNode);
        if(rs->size()>0){
            segFolderNode=rs->GetElement(0);
            if(segFolderNode.IsNotNull())
                segFolderNode->SetFloatProperty("point.displaysize",size);
        }
    }

    for(int i=0;i<selectedNodes.size();i++)
    {
        mitk::DataNode::Pointer node = selectedNodes[i];
        if(!isGroup->CheckNode(node))
            continue;

        node->SetFloatProperty("point.displaysize",size);
        svContourGroup* group=dynamic_cast<svContourGroup*>(node->GetData());
        if(group)
        {
            group->SetProp("point 2D display size",QString::number(size).toStdString());
            group->SetDataModified();
        }
    }

}

void svMLContourGroupPoint2DSizeAction::SetDataStorage(mitk::DataStorage *dataStorage)
{
    m_DataStorage = dataStorage;
}

void svMLContourGroupPoint2DSizeAction::SetFunctionality(berry::QtViewPart *functionality)
{
    m_Functionality=functionality;
}

