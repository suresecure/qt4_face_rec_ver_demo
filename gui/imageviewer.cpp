#include "gui/imageviewer.h"
//#include <QImageWriter>
//void ImageViewer::set_image(const QImage &img)
void ImageViewer::slotSetImage(const cv::Mat &frame)
{
//    const QImage img((const unsigned char*)frame.data, frame.cols, frame.rows, frame.step, //sizeof(QImage::Format_RGB888),
//                               QImage::Format_RGB888);
//    QTime now  = QTime::currentTime();
//    if (lastUpdateTime.msecsTo(now) < 20)
//    {
//        qDebug()<<"ImageViewer::set_image, time: "<<now.msecsTo(lastUpdateTime);
//        return;
//    }

    if (!image_.isNull())
        qDebug() << "The last paint has not been finished. Viewer dropped frame!";

//    image_ = img;
    this->frame_ = frame;
//    QImageWriter writer("c.jpg");
//    writer.write(image_);
//        qDebug() << "In ImageViewer: "<< image_.size()<<" bytes: "<<image_.byteCount()<<" bytes per line: "<<image_.bytesPerLine();
//    qDebug()<<"In ImageViewer: "<<frame.data[100];
//    if (image_.size() != size())
//        setFixedSize(image_.size());
    if (frame.size().height != size().width() || frame.size().height != size().width())
        setFixedSize(frame.size().width, frame.size().height);
    update();
//    lastUpdateTime = now;
}
