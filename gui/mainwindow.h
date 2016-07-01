#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
//#include <QMenuBar>
//#include <QMenu>
//#include <QAction>
#include "gui/mainwidget.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    MainWidget* display_;
};

#endif // MAINWINDOW_H
