#include "svMLContourGroupCreateAction.h"

#include <mitkNodePredicateDataType.h>

//#include <QmitkDataManagerView.h>

svMLContourGroupCreateAction::svMLContourGroupCreateAction()
    : m_ContourGroupCreateWidget(NULL)
    , m_Functionality(NULL)
{
}

svMLContourGroupCreateAction::~svMLContourGroupCreateAction()
{
    if(m_ContourGroupCreateWidget)
        delete m_ContourGroupCreateWidget;
}

void svMLContourGroupCreateAction::Run(const QList<mitk::DataNode::Pointer> &selectedNodes)
{
    mitk::DataNode::Pointer selectedNode = selectedNodes[0];

    mitk::NodePredicateDataType::Pointer isSegFolder = mitk::NodePredicateDataType::New("svSegmentationFolder");

    if(!isSegFolder->CheckNode(selectedNode))
    {
        return;
    }

    try
    {
//        if(!m_Functionality)
//            return;

//        QmitkDataManagerView* dmView=dynamic_cast<QmitkDataManagerView*>(m_Functionality);

//        if(!dmView)
//            return;

//        mitk::IRenderWindowPart* renderWindowPart = dmView->GetRenderWindowPart();

//        if(!renderWindowPart)
//            return;

//        mitk::SliceNavigationController* timeNavigationController=renderWindowPart->GetTimeNavigationController();
        int timeStep=0;
//        if(timeNavigationController)
//        {
//            timeStep=timeNavigationController->GetTime()->GetPos();
//        }

        if(m_ContourGroupCreateWidget)
        {
            delete m_ContourGroupCreateWidget;
        }

        m_ContourGroupCreateWidget=new svMLContourGroupCreate(m_DataStorage, selectedNode, timeStep);
        m_ContourGroupCreateWidget->show();
        m_ContourGroupCreateWidget->SetFocus();


    }
    catch(...)
    {
        MITK_ERROR << "Contour Group Creation Error!";
    }
}


void svMLContourGroupCreateAction::SetDataStorage(mitk::DataStorage *dataStorage)
{
    m_DataStorage = dataStorage;
}

void svMLContourGroupCreateAction::SetFunctionality(berry::QtViewPart *functionality)
{
    m_Functionality=functionality;
}

