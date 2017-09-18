#ifndef SVMLSEGMENTATIONPLUGINACTIVATOR_H
#define SVMLSEGMENTATIONPLUGINACTIVATOR_H

#include <ctkPluginActivator.h>

class svMLSegmentationPluginActivator :
        public QObject, public ctkPluginActivator
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "org_sv_gui_qt_ml")
    Q_INTERFACES(ctkPluginActivator)

public:

//    static ctkPluginContext* GetContext();
//    static svMLSegmentationPluginActivator* GetInstance();

    void start(ctkPluginContext* context) override;
    void stop(ctkPluginContext* context) override;

private:
//    static svMLSegmentationPluginActivator* m_Instance;
//    static ctkPluginContext* m_Context;

};

#endif // SVMLSEGMENTATIONPLUGINACTIVATOR_H
