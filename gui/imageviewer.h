#ifndef IMAGEVIEWER_H
#define IMAGEVIEWER_H

#include <QWidget>
#include <QImage>
#include <QPainter>
#include <QDebug>
//#include <QTime>
#include <opencv2/opencv.hpp>

class ImageViewer : public QWidget
{
    Q_OBJECT
    QImage image_;

    void paintEvent(QPaintEvent *)
    {
//        if (image_.size().isEmpty())
//            return;
//        cv::Mat mat = cv::Mat(image_.height(), image_.width(), CV_8UC3, (void*)image_.constBits(), image_.bytesPerLine());
//        cv::imwrite("b.jpg", mat);
        const QImage img((const unsigned char*)frame_.data, frame_.cols, frame_.rows, frame_.step,
                                   QImage::Format_RGB888);
        image_ = img;
        QPainter p(this);
        p.drawImage(0, 0, image_);
//        qDebug() << "In paintEvent: "<< char(image_.bits()[300*1920+1]);
        image_ = QImage();
        frame_.release();
    }

public:
    explicit ImageViewer(QWidget *parent = 0) : QWidget(parent)
    {
        setAttribute(Qt::WA_OpaquePaintEvent);
//        lastUpdateTime = QTime::currentTime();
//        image_ = QImage();
    }

signals:

public slots:
//    void set_image(const QImage & img);
    void slotSetImage(const cv::Mat & frame);

private:
//    QTime lastUpdateTime;
    cv::Mat frame_;
};

#endif // IMAGEVIEWER_H
