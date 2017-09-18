#ifndef SVMLLEVELSET2DWIDGET_H
#define SVMLLEVELSET2DWIDGET_H

#include "svSegmentationUtils.h"

#include <QPushButton>
#include <QWidget>


namespace Ui {
class svMLLevelSet2DWidget;
}

class svMLLevelSet2DWidget : public QWidget
{
    Q_OBJECT

public:

    svMLLevelSet2DWidget();

    svMLLevelSet2DWidget(QWidget* parent);

    virtual ~svMLLevelSet2DWidget();

    svSegmentationUtils::svLSParam GetLSParam();

    QPushButton* GetDoButton();

public slots:


protected:

  Ui::svMLLevelSet2DWidget *ui;

};

#endif // SVMLLEVELSET2DWIDGET_H
