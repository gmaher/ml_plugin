#ifndef SVMLLOFTPARAMWIDGET_H
#define SVMLLOFTPARAMWIDGET_H

#include <org_sv_gui_qt_segmentation_Export.h>

#include "svContourGroup.h"

#include <QWidget>

namespace Ui {
class svMLLoftParamWidget;
}

class SV_QT_SEGMENTATION svMLLoftParamWidget : public QWidget
{
    Q_OBJECT

public:
    explicit svMLLoftParamWidget(QWidget *parent = 0);
    ~svMLLoftParamWidget();

    void UpdateGUI(svLoftingParam* param);

    void UpdateParam(svLoftingParam* param);

    void SetButtonGroupVisible(bool visible);

    //private:
    Ui::svMLLoftParamWidget *ui;

public slots:

    void SelectionChanged(const QString &text);


};

#endif // SVMLLOFTPARAMWIDGET_H
