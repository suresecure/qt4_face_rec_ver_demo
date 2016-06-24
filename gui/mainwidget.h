#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QWidget>
#include <QThread>
#include <QLabel>
#include <QStringList>
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

private:
    FaceProcessor* face_processor_;
    Camera* camera_;
    QThread faceProcessThread_;
    QThread cameraThread_;
    // Camera view
    ImageViewer* image_viewer_;
//    // Show result faces
//    ImageViewer* result_viewer_[];
//    QLabel* result_rank_[];
//    QLabel* result_name_[];
//    QLabel* result_sim_[];

//    cv::Mat result_faces_[];

    // Show result faces
    ImageViewer* result_viewer_[RESULT_FACES_NUM];
    QLabel* result_rank_[RESULT_FACES_NUM];
    QLabel* result_name_[RESULT_FACES_NUM];
    QLabel* result_sim_[RESULT_FACES_NUM];

    cv::Mat result_faces_[RESULT_FACES_NUM];

signals:
    void sigResultFaceMatReady(const cv::Mat & frame);

public slots:
    void slotResultFacesReady(const QList<cv::Mat> result_faces,
                              const QStringList result_names,
                              const QStringList result_phones,
                              const QList<float> result_sim);
};

#endif // MAINWIDGET_H
