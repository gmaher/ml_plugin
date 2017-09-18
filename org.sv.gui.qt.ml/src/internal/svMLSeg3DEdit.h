#ifndef svSeg3DUtils_H
#define svSeg3DUtils_H

#include "svMitkSeg3D.h"
#include "svMitkSeg3DDataInteractor.h"

#include <QmitkFunctionality.h>

#include <vtkImageData.h>

#include <ctkRangeWidget.h>

namespace Ui {
class svMLSeg3DEdit;
}

class svMLSeg3DEdit : public QmitkFunctionality
{
    Q_OBJECT

public:

    enum SegmentationMethod {LEVELSET_METHOD, THRESHOLD_METHOD, REGION_GROWING_METHOD};

    static const QString EXTENSION_ID;

    svMLSeg3DEdit();

    virtual ~svMLSeg3DEdit();

public slots:

    void CreateByCollidingFronts();

    void SetSeedVisibility( bool checked = false );

    void ClearAll();

public:

//    int GetTimeStep();

    virtual void CreateQtPartControl(QWidget *parent) override;

    virtual void OnSelectionChanged(std::vector<mitk::DataNode*> nodes) override;

    virtual void NodeChanged(const mitk::DataNode* node) override;

    virtual void NodeAdded(const mitk::DataNode* node) override;

    virtual void NodeRemoved(const mitk::DataNode* node) override;

//    virtual void Activated() override;

//    virtual void Deactivated() override;

    virtual void Visible() override;

    virtual void Hidden() override;

//    bool IsExclusiveFunctionality() const override;

protected:

    QWidget* m_Parent;

    Ui::svMLSeg3DEdit *ui;

    vtkImageData* m_VtkImage;

    svMitkSeg3D* m_MitkSeg3D;

    mitk::DataNode::Pointer m_MitkSeg3DNode;

    svMitkSeg3DDataInteractor::Pointer m_DataInteractor;

    QmitkStdMultiWidget* m_DisplayWidget;
};

#endif // svSeg3DUtils_H
