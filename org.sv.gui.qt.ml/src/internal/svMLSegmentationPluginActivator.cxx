#include "svMLSegmentationPluginActivator.h"
#include "svMLContourGroupCreateAction.h"
#include "svMLSeg3DCreateAction.h"
#include "svMLSegmentationLegacyLoadAction.h"
#include "svMLSegmentationLegacySaveAction.h"
#include "svMLSegmentationLoadAction.h"
#include "svMLContourGroupPoint2DSizeAction.h"
#include "svMLContourGroupPoint3DSizeAction.h"
#include "svMLSeg2DEdit.h"
#include "svMLSeg3DEdit.h"

//svMLSegmentationPluginActivator* svMLSegmentationPluginActivator::m_Instance = nullptr;
//ctkPluginContext* svMLSegmentationPluginActivator::m_Context = nullptr;

void svMLSegmentationPluginActivator::start(ctkPluginContext* context)
{
//    m_Instance = this;
//    m_Context = context;

    BERRY_REGISTER_EXTENSION_CLASS(svMLContourGroupCreateAction, context)
    BERRY_REGISTER_EXTENSION_CLASS(svMLSeg3DCreateAction, context)
    BERRY_REGISTER_EXTENSION_CLASS(svMLSegmentationLegacyLoadAction, context)
    BERRY_REGISTER_EXTENSION_CLASS(svMLSegmentationLegacySaveAction, context)
    BERRY_REGISTER_EXTENSION_CLASS(svMLSeg2DEdit, context)
    BERRY_REGISTER_EXTENSION_CLASS(svMLSeg3DEdit, context)
    BERRY_REGISTER_EXTENSION_CLASS(svMLContourGroupPoint2DSizeAction, context)
    BERRY_REGISTER_EXTENSION_CLASS(svMLContourGroupPoint3DSizeAction, context)
    BERRY_REGISTER_EXTENSION_CLASS(svMLSegmentationLoadAction, context)
}

void svMLSegmentationPluginActivator::stop(ctkPluginContext* context)
{
//    Q_UNUSED(context)

//    m_Context = nullptr;
//    m_Instance = nullptr;
}

//ctkPluginContext* svMLSegmentationPluginActivator::GetContext()
//{
//  return m_Context;
//}

//svMLSegmentationPluginActivator* svMLSegmentationPluginActivator::GetInstance()
//{
//    return m_Instance;
//}

