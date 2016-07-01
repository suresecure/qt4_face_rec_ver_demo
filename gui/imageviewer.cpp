#include "gui/imageviewer.h"
//#include <QImageWriter>
//#include <QTime>

ImageViewer::ImageViewer( QWidget *parent, int handle ) : QWidget(parent)
{
    setAttribute(Qt::WA_OpaquePaintEvent);
    handle_ = handle;
//        lastUpdateTime = QTime::currentTime();
//        image_ = QImage();
}

//void ImageViewer::set_image(const QImage &img)
void ImageViewer::slotSetImage(const cv::Mat &frame, int handle)
{
//    const QImage img((const unsigned char*)frame.data, frame.cols, frame.rows, frame.step, //sizeof(QImage::Format_RGB888),
//                               QImage::Format_RGB888);
//    QTime now  = QTime::currentTime();
//    if (lastUpdateTime.msecsTo(now) < 20)
//    {
//        qDebug()<<"ImageViewer::set_image, time: "<<now.msecsTo(lastUpdateTime);
//        return;
//    }

    if (handle_ != handle)
        return;

    if (!image_.isNull())
        qDebug() << "The last paint has not been finished. Viewer dropped frame!";

//    image_ = img;
    this->frame_ = frame;
//    QImageWriter writer("c.jpg");
//    writer.write(image_);
//        qDebug() << "In ImageViewer: "<< image_.size()<<" bytes: "<<image_.byteCount()<<" bytes per line: "<<image_.bytesPerLine();
//    qDebug()<<"In ImageViewer: "<<frame_.data[99];
//    if (image_.size() != size())
//        setFixedSize(image_.size());
    if (frame.size().height != size().width() || frame.size().height != size().width())
        setFixedSize(frame.size().width, frame.size().height);
    update();
//    lastUpdateTime = now;
}

void ImageViewer::paintEvent(QPaintEvent *)
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
