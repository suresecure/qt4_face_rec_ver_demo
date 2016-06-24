#ifndef FACE_PROCESSOR_H
#define FACE_PROCESSOR_H

#include <QObject>
#include <QBasicTimer>
#include <QTimerEvent>
#include <QDir>
#include <QDebug>
#include <QImage>
//#include <QString>
#include <QStringList>
#include <QResource>
//#include <QVector>
//#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "face_repository.hpp"
#include "face_align.h"

namespace fs = ::boost::filesystem;
using namespace cv;
using namespace std;
using namespace face_rec_srzn;

Q_DECLARE_METATYPE(cv::Mat)
Q_DECLARE_METATYPE(QList<cv::Mat>)
Q_DECLARE_METATYPE(QList<float>)
class FaceProcessor : public QObject
{
    Q_OBJECT

public:
    FaceProcessor(QObject *parent=0, bool processAll = false);
    ~FaceProcessor();

    // Process all images camera captured or not.
    void setProcessAll(bool all);
//    static void matDeleter(void* mat);

//    void init(QString );

    // Get the class name of an image. JUST DEMO!
    // Now we assume that all face images of the same person store
    // in the same folder. So an image's class name is the parent fold name.
    string get_class_name(const string path) {
      return fs::canonical(path).parent_path().filename().string();
    }

    enum PROCESS_STATE{RECOGNITION = 0, REGISTER, VERIFICATION};

signals:
    void sigDisplayImageReady(const cv::Mat& frame);
    void sigResultFacesReady(const QList<cv::Mat> result_faces,
                             const QStringList result_names,
                             const QStringList result_phones,
                             const QList<float> result_sim);
    void sigNoFaceDetect();
public slots:
    void slotProcessFrame(const cv::Mat& frame);

private:
    // To process camera capture images.
    QBasicTimer timer_;
    cv::Mat frame_;     // Current camera frame.
    bool processAll_;
    void process(cv::Mat frame);
    void queue(const cv::Mat & frame);
    void timerEvent(QTimerEvent* ev);

    // Face detection, recognition, repository models' path.
    string caffe_model_folder_;
    string bayesian_model_path_;
    string dlib_face_model_path_;
    string face_repo_path_;

    // Face recognition, alignment, repository classes.
    LightFaceRecognizer * recognizer_;
    FaceAlign * face_align_;
    FaceRepo * face_repo_;

    // Face repository.
    vector<string> face_image_path_;  // All image paths, used by "FaceRepo".
    vector <vector<string> > person_image_path_;  //Face image path for each person.
    vector <string> person_; // Person names.
    void dispatchImage2Person();  // Dispatch FaceRepo's image paths to each person.

    PROCESS_STATE work_state_;

    // Empty (white) images.
//    QList<cv::Mat> empty_result_;
    cv::Mat empty_result_;

    // Face recognition parameters
    int  face_rec_knn_; // Size of return knn;
    float face_rec_th_dist_; // Distance threshold for same person.
    int face_rec_th_n_; // Least number of retrieved knn with same label.

    // Face register parameter
    int face_reg_num_capture_; // Number of faces captured in register.

    // Face verification parameter
    int  face_ver_knn_; // Size of return knn;
    float face_ver_th_dist_; // Distance threshold for same person.
    int face_ver_th_n_; // Least number of retrieved knn with same label.
};

#endif // FACE_PROCESSOR_H
