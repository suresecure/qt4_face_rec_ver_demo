#include "gui/mainwindow.h"
//#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    display_ = new MainWidget(this);
    setCentralWidget(display_);
}

MainWindow::~MainWindow()
{
}

