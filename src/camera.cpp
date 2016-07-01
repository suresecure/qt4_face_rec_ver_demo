#include "camera.h"

Camera::~Camera()
{
}

void Camera::slotRun()
{
    if (!video_capture_ || !using_video_camera_)
    {
        if (using_video_camera_)
            video_capture_.reset(new cv::VideoCapture(camera_index_));
        else
            video_capture_.reset(new cv::VideoCapture(video_file_name_));
    }
    if (video_capture_->isOpened())
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
    if (!video_capture_->read(frame)) // Blocks until a new frame is ready
    {
        timer_.stop();
        return;
    }
    emit sigMatReady(frame);
}

void Camera::slotUsingVideoCamera(bool value)
{
    using_video_camera_ = value;
}

void Camera::slotCameraIndex(int index)
{
    camera_index_ = index;
}

void Camera::slotVideoFileName(QString fileName)
{
    video_file_name_ = fileName.toStdString().c_str();
}
