#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>

#include <QObject>
#include <QScopedPointer>
#include <QTimerEvent>
#include <stdio.h>
#include <iostream>
#include <QImage>
#include <QBasicTimer>
#include <QDebug>

class Camera : public QObject
{
    Q_OBJECT
    QScopedPointer<cv::VideoCapture> video_capture_;
    QBasicTimer timer_;
    bool run_;
    bool using_video_camera_;
    int camera_index_;
    cv::String video_file_name_;

public:
    Camera(int camera_index=0, QObject* parent=0) : QObject(parent)
    {
        camera_index_ = camera_index;
        using_video_camera_ = true;
    }

    ~Camera();
    QImage convertToQImage( cv::Mat frame );


public slots:
    void slotRun();
    void slotCameraIndex(int index);
    void slotVideoFileName(QString fileName);
    void slotUsingVideoCamera(bool value);
    void slotStopped();

signals:
    void sigStarted();
    void sigMatReady(const cv::Mat &);

private:
    void timerEvent(QTimerEvent * ev);
};
