#include "svMLLevelSet2DWidget.h"
#include "ui_svMLLevelSet2DWidget.h"

svMLLevelSet2DWidget::svMLLevelSet2DWidget() :
    ui(new Ui::svMLLevelSet2DWidget)
{
    ui->setupUi(this);
    ui->doButton->hide();
}

svMLLevelSet2DWidget::svMLLevelSet2DWidget(QWidget* parent) :
    QWidget(parent),
    ui(new Ui::svMLLevelSet2DWidget)
{
    ui->setupUi(this);
    ui->doButton->hide();
}

svMLLevelSet2DWidget::~svMLLevelSet2DWidget()
{
    delete ui;
}

svSegmentationUtils::svLSParam svMLLevelSet2DWidget::GetLSParam()
{
    svSegmentationUtils::svLSParam lsparam;

    lsparam.radius=ui->lineRadius->text().trimmed().toDouble();

    lsparam.sigmaFeat1=ui->lineFeat1->text().trimmed().toDouble();
    lsparam.sigmaAdv1=ui->lineAdv1->text().trimmed().toDouble();
    lsparam.kc=ui->lineKthr->text().trimmed().toDouble();
    lsparam.expFactorRising=ui->lineRise->text().trimmed().toDouble();
    lsparam.expFactorFalling=ui->lineFall->text().trimmed().toDouble();
    lsparam.maxIter1=ui->lineMaxIters1->text().trimmed().toInt();
    lsparam.maxErr1=ui->lineMaxErr1->text().trimmed().toDouble();

    lsparam.sigmaFeat2=ui->lineFeat2->text().trimmed().toDouble();
    lsparam.sigmaAdv2=ui->lineAdv2->text().trimmed().toDouble();
    lsparam.kupp=ui->lineKupp->text().trimmed().toDouble();
    lsparam.klow=ui->lineKlow->text().trimmed().toDouble();
    lsparam.maxIter2=ui->lineMaxIters2->text().trimmed().toInt();
    lsparam.maxErr2=ui->lineMaxErr2->text().trimmed().toDouble();

    return lsparam;
}

QPushButton* svMLLevelSet2DWidget::GetDoButton()
{
    return ui->doButton;
}

