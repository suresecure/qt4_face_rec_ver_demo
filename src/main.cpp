#include "gui/mainwindow.h"
#include "gui/mainwidget.h"
#include <QApplication>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
//    MainWidget w;
//    w.show();
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType< QList<cv::Mat> >("QList<cv::Mat>");
    qRegisterMetaType< QList<float> >("QList<float>");

    return a.exec();
}
