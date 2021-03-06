#ifndef SVMLSEG2DEDIT_H
#define SVMLSEG2DEDIT_H

#include "svPath.h"
#include "svSegmentationUtils.h"
#include "svContourGroup.h"
#include "svContourModel.h"
#include "svContourGroupDataInteractor.h"
#include "svMLContourGroupCreate.h"
#include "svContourModelThresholdInteractor.h"

#include "svResliceSlider.h"
#include "svMLLevelSet2DWidget.h"
#include "svMLLoftParamWidget.h"

#include <QmitkFunctionality.h>
#include <QmitkSliceWidget.h>
#include <QmitkSliderNavigatorWidget.h>
#include <QmitkStepperAdapter.h>

#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <mitkSurface.h>
#include <mitkPointSet.h>
#include <mitkDataInteractor.h>
#include <mitkImage.h>

#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkTransform.h>

#include <ctkSliderWidget.h>

#include <QSlider>
#include <QPushButton>
#include <QWidget>

namespace Ui {
class svMLSeg2DEdit;
}

class svMLSeg2DEdit : public QmitkFunctionality
{
    Q_OBJECT

public:

    enum SegmentationMethod {LEVELSET_METHOD, THRESHOLD_METHOD, REGION_GROWING_METHOD, NN2D_METHOD};

    static const QString EXTENSION_ID;

    svMLSeg2DEdit();

    virtual ~svMLSeg2DEdit();

public slots:

    void CreateNNContour();

    void CreateLSContour();

    void CreateThresholdContour();

    void CreateCircle();

    void CreateEllipse();

    void CreateSplinePoly();

    void CreatePolygon();

    void UpdateContourList();

    void InsertContour(svContour* contour, int contourIndex);

    void InsertContourByPathPosPoint(svContour* contour);

    void SetContour(int contourIndex, svContour* newContour);

    void RemoveContour(int contourIndex);

    void DeleteSelected();

    void SelectContour();

    void SelectContour(const QModelIndex & idx);

    void ClearAll();

    void UpdatePreview();

    void FinishPreview();

    void SmoothSelected();

    void CreateContours(SegmentationMethod method);

    void SetSecondaryWidgetsVisible(bool visible);

    std::vector<int> GetBatchList();

    svContour* PostprocessContour(svContour* contour);

    double GetVolumeImageSpacing();

    void LoftContourGroup();

    void ShowLoftWidget();

    void UpdateContourGroupLoftingParam();

    void OKLofting();

    void ApplyLofting();

    void HideLoftWidget();

    void ContourChangingOn();

    void ContourChangingOff();

    void SelectContour3D();

    void ResetGUI();

    void UpdatePathResliceSize(double newSize);

    void ManualContextMenuRequested();

    void ManualCircleContextMenuRequested(const QPoint&);
    void ManualEllipseContextMenuRequested(const QPoint&);
    void ManualSplinePolyContextMenuRequested(const QPoint&);
    void ManualPolygonContextMenuRequested(const QPoint&);

    void CreateManualCircle( bool checked = false );

    void CreateManualEllipse( bool checked = false );

    void CreateManualSplinePoly( bool checked = false );

    void CreateManualPolygon( bool checked = false );

    void CreateManualPolygonType(bool spline);

    void CopyContour();

    void PasteContour();

    void NewGroup();

    void ShowPath(bool checked = false);

    void UpdatePathPoint(int pos);

public:

    void SelectContour(int index);

    int GetTimeStep();

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

    void PreparePreviewInteraction(QString method);

    void QuitPreviewInteraction();

protected:

    bool eventFilter(QObject *obj, QEvent *ev);

    QWidget* m_Parent;

    QWidget* m_CurrentParamWidget;

    QWidget* m_CurrentSegButton;

    svMLLevelSet2DWidget* m_LSParamWidget;

    svMLLoftParamWidget* m_LoftWidget;

    mitk::Image* m_Image;

    cvStrPts* m_cvImage;

    Ui::svMLSeg2DEdit *ui;

    svContourGroup* m_ContourGroup;

    svPath* m_Path;

    mitk::DataNode::Pointer m_PathNode;

    mitk::DataNode::Pointer m_ContourGroupNode;

    mitk::DataNode::Pointer m_GroupFolderNode;

    svContourGroupDataInteractor::Pointer m_DataInteractor;

    long m_ContourGroupChangeObserverTag;

    mitk::DataNode::Pointer m_LoftSurfaceNode;

    mitk::Surface::Pointer m_LoftSurface;

    bool groupCreated=false;

    svContourModelThresholdInteractor::Pointer m_PreviewDataNodeInteractor;

    mitk::DataNode::Pointer m_PreviewDataNode;

    long m_PreviewContourModelObserverFinishTag;
    long m_PreviewContourModelObserverUpdateTag;

    svContourModel::Pointer m_PreviewContourModel;

    long m_StartLoftContourGroupObserverTag;
    long m_StartLoftContourGroupObserverTag2;
    long m_StartChangingContourObserverTag;
    long m_EndChangingContourObserverTag;
    long m_SelectContourObserverTag;


    bool m_ContourChanging;

    QmitkStdMultiWidget* m_DisplayWidget;

    std::vector<svPathElement::svPathPoint> m_PathPoints;

    QMenu* m_ManualMenu;

    svContour* m_CopyContour;

    svMLContourGroupCreate* m_ContourGroupCreateWidget;

    bool m_UpdatingGUI;
};

#endif // SVMLSEG2DEDIT_H
