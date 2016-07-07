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
#include "settings.h"

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

    // Get the class name of an image. JUST DEMO!
    // Now we assume that all face images of the same person store
    // in the same folder. So an face's person name is the parent fold name.
    string getPersonName(const string path) {
      return fs::canonical(path).parent_path().filename().string();
    }

    // Three work states. Default state is face recognition.
    enum PROCESS_STATE{STATE_DEFAULT = 0, STATE_VERIFICATION, STATE_REGISTER};

signals:
    void sigDisplayImageReady(const cv::Mat& frame);
    void sigResultFacesReady(const QList<cv::Mat> result_faces,
                             const QStringList result_names,
//                             const QStringList result_phones,
                             const QList<float> result_sim);
//    void sigNoFaceDetect();
    void sigCaptionUpdate(QString caption);
    void sigRegisterDone();
    void sigVerificationDone();

public slots:
    void slotProcessFrame(const cv::Mat& frame);
    void slotRegister(bool start, QString name);
    void slotVerification(bool start, QString name);

private:
    // To process camera capture images.
    void process(cv::Mat frame);
    void queue(const cv::Mat & frame);
    void timerEvent(QTimerEvent* ev);

    bool faceRepoInit();

    // Dispatch FaceRepo's image paths to each person.
    void dispatchImage2Person();

    // Do face recognition
    void faceRecognition( const cv::Mat & query, const int knn, const float th_dist,
                          map<float, pair<int, string> > & combined_result,
                          map <string, string> & example_face);

    // For face verification or register
    bool checkFacePose(const Mat & feature, const Mat & H, const Mat & inv_H); // Check if the current face is valid.
    void verAndSelectFace(const Mat & face, const Mat & feature, const Mat & H, const Mat & inv_H); // Select a face.
    void cleanSelectedFaces(); // Clean verification or register statement.
//    void faceRegister();


private:
    // Timer to save face repository.
    QBasicTimer save_timer_;
    bool face_repo_is_dirty_;

    // To process camera captured images.
    QBasicTimer frame_timer_;
    cv::Mat frame_;     // Current camera frame.
    bool process_all_;  // Process every frame captured by the camera?

    // Face detection, recognition, repository models' path.
    string caffe_model_folder_;
    string bayesian_model_path_;
    string dlib_face_model_path_;
    string face_repo_path_;
    string face_image_home_; // Root directory to store face images.

    // Face recognition, alignment, repository classes.
    LightFaceRecognizer * recognizer_;
    FaceAlign * face_align_;
    FaceRepo * face_repo_;

    // Face repository.
    vector<string> face_image_path_;  // All image paths, used by "FaceRepo".
    vector <vector<string> > person_image_path_;  //Face image path for each person.
    vector <string> person_; // Person names.

    // Working statement: recognition, verification or register.
    PROCESS_STATE work_state_;

    // Empty (white) result images.
    cv::Mat empty_result_;

    // Face recognition parameters
    int  face_rec_knn_; // Size of return knn;
    float face_rec_th_dist_; // Distance threshold for same person.
    int face_rec_th_n_; // Least number of retrieved knn with same label.

    // Face verification parameter
    int  face_ver_knn_; // Size of return knn;
    float face_ver_th_dist_; // Distance threshold for same person.
    int face_ver_th_n_; // Least number of retrieved knn with same label.
    int face_ver_sample_num_; // Number of sample faces to compare directly.
    int face_ver_num_; // Number of faces to be checked  in verification.
    int face_ver_valid_num_; // Minimun number of  accepted faces to verificate a person.
    cv::Mat face_ver_target_samlpe_; // A sample face of verification target person.

    // Face register parameter
    int face_reg_num_; // Number of faces needed in register.
    string face_reg_ver_name_; // Person name in face register of verification.
    bool face_reg_need_ver_; // Need to verificate current person before register because the name already exist.

    // Select faces in person verification or register.
    // We should use faces with different poses for robust reason.
    int selected_faces_num_; // Number of selected faces.
    vector<cv::Mat> selected_face_H_;  // Affine matrix in alignment of selected faces.
    vector<cv::Mat> selected_face_inv_H_;  // Inverse affine matrix in alignment of selected faces.
    float  pose_min_dist_; // Minimum pose distance.
    vector<cv::Mat> selected_face_aligned_;  // Selected faces (aligned).
    vector<cv::Mat> selected_face_feature_;  // Feature of selected faces.
    vector<bool> selected_face_ver_valid_; // Is the selected face pass the verification.
    int selected_face_ver_valid_num_; // Number of selected face that passed the verification.
    float feature_min_dist_;  // Minimun feature distance to make different pose.
    float feature_max_dist_; // Maximum feature distance to assure same person in front of the camera.
};

#endif // FACE_PROCESSOR_H
