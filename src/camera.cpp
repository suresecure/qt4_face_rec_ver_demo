#include "camera.h"

Camera::~Camera()
{
}

void Camera::slotRun()
{
    // TODO: clean up. Would be nice not to have nested `if` statements
    if (!videoCapture_ or !usingVideoCamera_)
    {
        if (usingVideoCamera_)
            videoCapture_.reset(new cv::VideoCapture(cameraIndex_));
        else
            videoCapture_.reset(new cv::VideoCapture(videoFileName_));
    }
    if (videoCapture_->isOpened())
    {
        timer_.start(0, this);
        emit sigStarted();
    }
}

void Camera::slotStopped()
{
    timer_.stop();
}

void Camera::timerEvent(QTimerEvent *ev)
{
    if (ev->timerId() != timer_.timerId())
        return;
    cv::Mat frame;
    if (!videoCapture_->read(frame)) // Blocks until a new frame is ready
    {
        timer_.stop();
        return;
    }
    emit sigMatReady(frame);
}

void Camera::slotUsingVideoCamera(bool value)
{
    usingVideoCamera_ = value;
}

void Camera::slotCameraIndex(int index)
{
    cameraIndex_ = index;
}

void Camera::slotVideoFileName(QString fileName)
{
    videoFileName_ = fileName.toStdString().c_str();
}
