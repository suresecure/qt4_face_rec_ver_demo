#include "gui/imageviewer.h"
//#include <QImageWriter>
//#include <QTime>

ImageViewer::ImageViewer( QWidget *parent, int handle ) : QWidget(parent)
{
    setAttribute(Qt::WA_OpaquePaintEvent);
    _handle = handle;
}

void ImageViewer::slotSetImage(const cv::Mat &frame, int handle)
{

    if (_handle != handle)
        return;

    if (!_image.isNull())
        qDebug() << "The last paint has not been finished. Viewer dropped frame!";

    this->_frame = frame;
    if (frame.size().height != size().width() || frame.size().height != size().width())
        setFixedSize(frame.size().width, frame.size().height);
    update();
}

void ImageViewer::paintEvent(QPaintEvent *)
{
    const QImage img((const unsigned char*)_frame.data, _frame.cols, _frame.rows, _frame.step,
                               QImage::Format_RGB888);
    _image = img;
    QPainter p(this);
    p.drawImage(0, 0, _image);
    _image = QImage();
    _frame.release();
}
