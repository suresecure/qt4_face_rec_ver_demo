#include "mainwidget.h"

//#include <iostream>
//#include <stdio.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
//#include <QGridLayout>
//#include <QPushButton>
//#include <QRadioButton>
//#include <QLineEdit>
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
    // Init state = recognition
    is_reg_ = false;
    is_ver_ = false;

    /*
     * GUI layout definition.
     */
    // Layout in the top for view
    QHBoxLayout * viewLayout = new QHBoxLayout();
    QVBoxLayout * cameraLayout = new QVBoxLayout();
    caption_ = new QLabel("Face Recognition", this);
    cameraLayout->addWidget(caption_);
    // Camera view
    image_viewer_ = new ImageViewer(this);
    cameraLayout->addWidget(image_viewer_);
    viewLayout->addLayout(cameraLayout);
    // Return face view
    QGridLayout * resultLayout = new QGridLayout();
    viewLayout->addLayout(resultLayout);
    for (int i = 0; i < RESULT_FACES_NUM; i++) {
        QVBoxLayout * oneResult = new QVBoxLayout();
        result_rank_[i] = new QLabel("#" + QString::number(i+1), this);
        result_name_[i] = new QLabel(this);
        result_sim_[i] = new QLabel(this);
        result_viewer_[i] = new ImageViewer(this, i);
        result_viewer_[i]->setFixedSize(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT);
        oneResult->addWidget(result_rank_[i], 0, Qt::AlignHCenter);
        oneResult->addWidget(result_viewer_[i], 0, Qt::AlignHCenter);
        oneResult->addWidget(result_name_[i]);
        oneResult->addWidget(result_sim_[i]);
        resultLayout->addLayout(oneResult, i/2, i%2);
    }

    // Layout in the bottom for input texts and push buttons.
    QHBoxLayout* controlLayout = new QHBoxLayout();
    // Input user name.
    QHBoxLayout* nameLayout = new QHBoxLayout();
    QLabel* nameLabel = new QLabel("Name or phone number (unique):", this);
    name_LineEdit_ = new QLineEdit(this);
    QValidator *nameValidator = new QRegExpValidator(QRegExp("^[a-zA-Z0-9]+$"), this );
    name_LineEdit_->setValidator(nameValidator);
    nameLayout->addWidget(nameLabel);
    nameLayout->addWidget(name_LineEdit_);
    // Two control push bottons.
    register_PushButton_ = new QPushButton("Register", this);
    verification_PushButton_ = new QPushButton("Verification", this);
    // Add all to the controlLayout
    controlLayout->addLayout(nameLayout);
//    controlLayout->addLayout(phoneLayout);
    controlLayout->addWidget(register_PushButton_);
    controlLayout->addWidget(verification_PushButton_);

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
    face_process_thread_.start();
    camera_thread_.start();
    camera_->moveToThread(&camera_thread_);
    face_processor_->moveToThread(&face_process_thread_);
    // Make connection
    image_viewer_->connect(face_processor_,
                           SIGNAL(sigDisplayImageReady(cv::Mat)),
                           SLOT(slotSetImage(cv::Mat)));
    face_processor_->connect(camera_, SIGNAL(sigMatReady( cv::Mat)),
                           SLOT(slotProcessFrame(cv::Mat)));
    face_processor_->connect(this, SIGNAL(sigRegister(bool, QString)), SLOT(slotRegister(bool, QString)));
    face_processor_->connect(this, SIGNAL(sigVerification(bool, QString)), SLOT(slotVerification(bool, QString)));
    this->connect(face_processor_,
                  SIGNAL(sigCaptionUpdate(QString)),
                  SLOT(slotCaptionUpdate(QString)));
    this->connect(face_processor_,
                  SIGNAL(sigResultFacesReady(QList<cv::Mat>, QStringList,  QList<float>)),
                  SLOT(slotResultFacesReady(QList<cv::Mat>, QStringList, QList<float>)));
    this->connect(face_processor_, SIGNAL(sigRegisterDone()), SLOT(slotRegisterDone()));
    this->connect(face_processor_, SIGNAL(sigVerificationDone()), SLOT(slotVerificationDone()));
    this->connect(register_PushButton_, SIGNAL(clicked()), SLOT(slotOnClickRegisterButton()));
    this->connect(verification_PushButton_, SIGNAL(clicked()), SLOT(slotOnClickVerificationButton()));
    for (int i = 0; i < RESULT_FACES_NUM; i++) {
        result_viewer_[i]->connect(this, SIGNAL(sigResultFaceMatReady(cv::Mat, int)),
                                   SLOT(slotSetImage(cv::Mat, int )));
    }
    // Start camera.
    QTimer::singleShot(0, camera_, SLOT(slotRun()));
}

void MainWidget::slotCaptionUpdate(QString caption)
{
    caption_->setText(caption);
}

MainWidget::~MainWidget()
{
    face_processor_->disconnect();
    camera_->disconnect();
    face_process_thread_.quit();
    camera_thread_.quit();
    face_process_thread_.wait();
    camera_thread_.wait();
    face_processor_->~FaceProcessor();
    camera_->~Camera();
}

void MainWidget::slotResultFacesReady(const QList<cv::Mat> result_faces,
                          const QStringList result_names,
//                          const QStringList result_phones,
                          const QList<float> result_sim)
{
    for (int i = 0; i < RESULT_FACES_NUM; i++) {
        result_name_[i]->setText(result_names[i]);
        if ( 0 == result_sim[i])
            result_sim_[i]->setText("");
        else
            result_sim_[i]->setText(QString("%1").arg(result_sim[i]));
//        qDebug()<<result_faces[i].data[99];
        result_faces_[i] = result_faces[i];
        emit sigResultFaceMatReady(result_faces_[i], i);
        //image_viewer_[i].slotSetImage(result_faces_[i]);
    }
//        cout<<"slotResultFacesReady: "<<int(resultFaces_[0].data[99])<<endl;
}


void MainWidget::slotOnClickRegisterButton()
{
    if ( is_reg_ )
    {
        is_reg_ = false;
        register_PushButton_->setText("Register");
        verification_PushButton_->setEnabled(true);
        name_LineEdit_->setEnabled(true);
        emit sigRegister(false);
    }
    else
    {
        if (name_LineEdit_->text().isEmpty())
            return;
        is_reg_ = true;
        register_PushButton_->setText("Reg Stop");
        verification_PushButton_->setEnabled(false);
        name_LineEdit_->setEnabled(false);
        emit sigRegister(true, name_LineEdit_->text());
    }
}

void MainWidget::slotOnClickVerificationButton()
{
    if ( is_ver_  )
    {
        is_ver_ = false;
        verification_PushButton_->setText("Verification");
        register_PushButton_->setEnabled(true);
        name_LineEdit_->setEnabled(true);
        emit sigVerification(false);
    }
    else
    {
        if (name_LineEdit_->text().isEmpty())
            return;
        is_ver_ = true;
        verification_PushButton_->setText("Ver Stop");
        register_PushButton_->setEnabled(false);
        name_LineEdit_->setEnabled(false);
        emit sigVerification(true, name_LineEdit_->text());
    }
}

void MainWidget::slotRegisterDone()
{
    register_PushButton_->setText("Continue");
}

void MainWidget::slotVerificationDone()
{
    verification_PushButton_->setText("Continue");
}
