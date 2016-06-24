#include "mainwidget.h"

//#include <iostream>
//#include <stdio.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QRadioButton>
#include <QLineEdit>
#include <QTimer>
#include <QRegExp>
#include <QValidator>
//#include <QComboBox>
//#include <QFileDialog>
#include <QDebug>

#include "gui/imageviewer.h"
#include "src/camera.h"
#include "src/face_processor.h"

MainWidget::MainWidget(QWidget *parent) :
    QWidget(parent)
{

    /*
     * GUI layout definition.
     */
    // Layout in the top for view
    QHBoxLayout * viewLayout = new QHBoxLayout();
    // Camera view
    image_viewer_ = new ImageViewer(this);
    viewLayout->addWidget(image_viewer_);
    // Return face view
    QGridLayout * resultLayout = new QGridLayout();
    viewLayout->addLayout(resultLayout);
    for (int i = 0; i < RESULT_FACES_NUM; i++) {
        QVBoxLayout * oneResult = new QVBoxLayout();
        result_rank_[i] = new QLabel("#" + QString::number(i+1), this);
        result_name_[i] = new QLabel(this);
        result_sim_[i] = new QLabel(this);
        result_viewer_[i] = new ImageViewer(this);
        result_viewer_[i]->setFixedSize(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT);
        oneResult->addWidget(result_rank_[i], 0, Qt::AlignHCenter);
        oneResult->addWidget(result_viewer_[i], 0, Qt::AlignHCenter);
        oneResult->addWidget(result_name_[i]);
        oneResult->addWidget(result_sim_[i]);
        resultLayout->addLayout(oneResult, i/2, i%2);
    }

    // Layout in the bottom for input texts and push buttons.
    QHBoxLayout* controlLayout = new QHBoxLayout();
   /* // Input user name
    QHBoxLayout* nameLayout = new QHBoxLayout();
    QLabel* nameLabel = new QLabel("Name:", this);
    QLineEdit* nameLineEdit = new QLineEdit(this);
    QValidator *nameValidator = new QRegExpValidator(QRegExp("^[a-zA-Z0-9]+$"), this );
    nameLineEdit->setValidator(nameValidator);
    nameLayout->addWidget(nameLabel);
    nameLayout->addWidget(nameLineEdit);
    // Input user phone number
    QHBoxLayout* phoneLayout = new QHBoxLayout();
    QLabel* phoneLabel = new QLabel("PhoneNo:", this);
    QLineEdit* phoneLineEdit = new QLineEdit(this);
    QValidator *phoneValidator = new QRegExpValidator(QRegExp("^[0-9]+$"), this );
    phoneLineEdit->setValidator(phoneValidator);
    phoneLayout->addWidget(phoneLabel);
    phoneLayout->addWidget(phoneLineEdit);
    */
    // Input user name
    QHBoxLayout* nameLayout = new QHBoxLayout();
    QLabel* nameLabel = new QLabel("Name or phone number (unique):", this);
    QLineEdit* nameLineEdit = new QLineEdit(this);
    QValidator *nameValidator = new QRegExpValidator(QRegExp("^[a-zA-Z0-9]+$"), this );
    nameLineEdit->setValidator(nameValidator);
    nameLayout->addWidget(nameLabel);
    nameLayout->addWidget(nameLineEdit);
    // Two control push bottons
    QPushButton *registerPushButton = new QPushButton("Register", this);
    QPushButton *verificationPushButton = new QPushButton("Verification", this);
    // Add all to the controlLayout
    controlLayout->addLayout(nameLayout);
//    controlLayout->addLayout(phoneLayout);
    controlLayout->addWidget(registerPushButton);
    controlLayout->addWidget(verificationPushButton);

    // Main layout
    QVBoxLayout * mainLayout = new QVBoxLayout(this);
    mainLayout->addLayout(viewLayout);
    mainLayout->addLayout(controlLayout);

    /*
     * Face recoginition and verification workflow.
     */
    camera_ = new Camera();
    face_processor_ = new FaceProcessor();
    face_processor_->setProcessAll(false);
    // Move camera and face detector to their independent threads
    faceProcessThread_.start();
    cameraThread_.start();
    camera_->moveToThread(&cameraThread_);
    face_processor_->moveToThread(&faceProcessThread_);
    // Make connection
    image_viewer_->connect(face_processor_,
                           SIGNAL(sigDisplayImageReady(cv::Mat)),
                           SLOT(slotSetImage(cv::Mat)));
    face_processor_->connect(camera_, SIGNAL(sigMatReady( cv::Mat)),
                           SLOT(slotProcessFrame(cv::Mat)));
    this->connect(face_processor_,
                  SIGNAL(sigResultFacesReady(QList<cv::Mat>, QStringList, QStringList, QList<float>)),
                  SLOT(slotResultFacesReady(QList<cv::Mat>, QStringList, QStringList, QList<float>)));
    for (int i = 0; i < RESULT_FACES_NUM; i++) {
        result_viewer_[i]->connect(this, SIGNAL(sigResultFaceMatReady(cv::Mat)),
                                   SLOT(slotSetImage(cv::Mat)));
    }
    // Start camera
    QTimer::singleShot(0, camera_, SLOT(slotRun()));

//    QObject::connect(cameraComboBox, SIGNAL(currentIndexChanged(int)),
//                     camera_, SLOT(cameraIndexSlot(int)));

//    QObject::connect(fileSelector, SIGNAL(clicked()),
//                     this,	SLOT(openFileDialog()));

//    QObject::connect(sourceSelector, SIGNAL(toggled(bool)),
//                     camera_, SLOT(usingVideoCameraSlot(bool)));

//    QObject::connect(this, SIGNAL(videoFileNameSignal(QString)),
//                     camera_, SLOT(videoFileNameSlot(QString)));

//    face_processor_->connect(this, SIGNAL(facecascade_name_signal(QString)),
//                     SLOT(facecascade_filename(QString)));
}

MainWidget::~MainWidget()
{
    face_processor_->~FaceProcessor();
    camera_->~Camera();
    faceProcessThread_.quit();
    cameraThread_.quit();
    faceProcessThread_.wait();
    cameraThread_.wait();
}

void MainWidget::slotResultFacesReady(const QList<cv::Mat> result_faces,
                          const QStringList result_names,
                          const QStringList result_phones,
                          const QList<float> result_sim)
{
//    result_faces_ = result_faces;
    for (int i = 0; i < RESULT_FACES_NUM; i++) {
        result_name_[i]->setText(result_names[i]);
        if ( 0 == result_sim[i])
            result_sim_[i]->setText("");
        else
            result_sim_[i]->setText(QString("%1").arg(result_sim[i]));
//        qDebug()<<result_faces[i].data[100];
        result_faces_[i] = result_faces[i];
        emit sigResultFaceMatReady(result_faces_[i]);
//        result_viewer_[i]->slotSetImage(result_faces[i]);
    }
}
