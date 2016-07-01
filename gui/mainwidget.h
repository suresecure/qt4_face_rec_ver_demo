#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QWidget>
#include <QThread>
#include <QLabel>
#include <QStringList>
#include <QPushButton>
#include <QLineEdit>
#include <opencv2/opencv.hpp>

#include "gui/imageviewer.h"
#include "src/camera.h"
#include "src/face_processor.h"
#include "src/settings.h"

class MainWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MainWidget(QWidget *parent = 0);
    ~MainWidget();

signals:
    void sigResultFaceMatReady(const cv::Mat & frame, int handle);
    void sigRegister(bool start, QString name = QString());
    void sigVerification(bool start, QString name = QString());

public slots:
    void slotResultFacesReady(const QList<cv::Mat> result_faces,
                              const QStringList result_names,
                              const QList<float> result_sim);
    void slotCaptionUpdate(QString caption);
    void slotRegisterDone();
    void slotVerificationDone();

private slots:
    void slotOnClickRegisterButton();
    void slotOnClickVerificationButton();

private:
    FaceProcessor* face_processor_;
    Camera* camera_;
    QThread face_process_thread_;
    QThread camera_thread_;

    QLabel* caption_;

    // Camera view
    ImageViewer* image_viewer_;

    // Show result faces
    ImageViewer* result_viewer_[RESULT_FACES_NUM];
    QLabel* result_rank_[RESULT_FACES_NUM];
    QLabel* result_name_[RESULT_FACES_NUM];
    QLabel* result_sim_[RESULT_FACES_NUM];
    cv::Mat result_faces_[RESULT_FACES_NUM];

    // Control buttons
    QPushButton *register_PushButton_;
    QPushButton *verification_PushButton_;
    bool is_reg_; // Doing register.
    bool is_ver_; // Doing verification.
    QLineEdit* name_LineEdit_;
};

#endif // MAINWIDGET_H
