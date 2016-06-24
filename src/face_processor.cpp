#include "face_processor.h"
//#include <QImageWriter>

#include <QCoreApplication>
//#include <flann/flann.hpp>
//#include <flann/io/hdf5.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <exception>

#include "face_recognition.hpp"
#include "face_repository.hpp"
#include "face_align.h"
#include "face_rep_utils.h"
#include "settings.h"

namespace fs = ::boost::filesystem;
using namespace cv;
using namespace std;
using namespace face_rec_srzn;

FaceProcessor::FaceProcessor(QObject *parent, bool processAll) : QObject(parent), processAll_(processAll)
{
    string appPath = QCoreApplication::applicationDirPath().toStdString();
    caffe_model_folder_ = fs::path(appPath + "/" + caffe_model_folder).string();
    bayesian_model_path_ = fs::path(appPath + "/" + bayesian_model_path).string();
    dlib_face_model_path_ = fs::path(appPath + "/" + dlib_face_model_path).string();
    face_repo_path_ = fs::path(appPath + "/" + face_repo_path).string();

    // Init
    recognizer_ = new LightFaceRecognizer(
        caffe_model_folder_,
        bayesian_model_path_,
        "prob", false);
    face_align_ = new FaceAlign(dlib_face_model_path_);
    work_state_ = RECOGNITION;

    Q_ASSERT(recognizer_ != NULL);
    Q_ASSERT(face_align_ != NULL);

    // Load face repository.
    face_repo_ = NULL;
    try
    {
        face_repo_ = new FaceRepo(*recognizer_);
        // Load face repository.
        face_repo_->Load(face_repo_path_);
        // Load all image paths.
        int N = face_repo_->GetValidFaceNum();
        fs::path save_image_path_file(face_repo_path_);
        save_image_path_file /= fs::path("dataset_file_path.txt");
        ifstream ifile(save_image_path_file.string().c_str());
        string line;
        for (int i = 0; i < N; i++) {
          getline(ifile, line);
          face_image_path_.push_back(line);
        }
        ifile.close();
    }
    catch(exception e)
    {
        qDebug()<<"Face repository does not exist or other error.";
        qDebug()<<e.what();
        QDir dir;
        dir.mkpath(QString::fromLocal8Bit(face_repo_path_.c_str()));
        face_image_path_.clear();
    }

    if ( 0 < face_image_path_.size() )
        dispatchImage2Person();
    /*// Print result to test dispatchImage2Person()
    cout<<"Person count: "<<person_.size()<<endl;
    for (int i = 0; i < person_.size(); i++)
    {
        cout<<person_[i]<<": "<<person_image_path_[i].size()<<endl;
        for (int j = 0; j < person_image_path_[i].size(); j++)
            cout<<j<<": "<<person_image_path_[i][j]<<endl;
    }*/

    // Empty (white) return images
//    for (int i = 0; i < RESULT_FACES_NUM; i++)
//    {
//        cv::Mat result(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT, CV_8UC3, Scalar(255, 255, 255));
//        empty_result_.append(result);
//    }
    empty_result_ = Mat(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT, CV_8UC3, Scalar(255, 255, 255));

    // Face recognition parameter
    face_rec_knn_ = FACE_REC_KNN;
    face_rec_th_dist_ = FACE_REC_TH_DIST;
    face_rec_th_n_ = FACE_REC_TH_N;
    // Person register parameter
    face_reg_num_capture_ = FACE_REG_NUM_CAPTURE;
    // Face verification parameter
    face_rec_knn_ = FACE_REC_KNN;
    face_rec_th_dist_ = FACE_REC_TH_DIST;
    face_rec_th_n_ = FACE_REC_TH_N;
}

FaceProcessor::~FaceProcessor()
{
    delete recognizer_;
    delete face_align_;
    if (NULL != face_repo_)
        delete face_repo_;
}

// Dispatch FaceRepo's image paths to each person.
void FaceProcessor::dispatchImage2Person()
{
    for (int i =  0 ; i < face_image_path_.size(); i++)
    {
        string person_name = get_class_name(face_image_path_[i]);
        vector<string>::iterator iter = find(person_.begin(), person_.end(), person_name);
        if ( iter == person_.end() )
        {
            person_.push_back(person_name);
            vector<string> person_images;
            person_images.push_back(face_image_path_[i]);
            person_image_path_.push_back(person_images);
        }
        else
        {
            int pos = iter - person_.begin();
            person_image_path_[pos].push_back(face_image_path_[i]);
        }
    }
}

void FaceProcessor::slotProcessFrame(const cv::Mat &frame)
{
    if (processAll_)
        process(frame);
    else
        queue(frame);
}

void FaceProcessor::setProcessAll(bool all)
{
    processAll_ = all;
}

void FaceProcessor::process(cv::Mat frame)
{
    QList<cv::Mat> result_faces;
    QStringList result_names;
    QStringList result_phones;
    QList<float> result_sim;

    Mat face_cropped;
    Rect rect_face_detected;
    rect_face_detected = detectAlignCropDlib(*face_align_, frame, face_cropped);
    if (0 == rect_face_detected.area())
    {// No face detected.
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        for (int i = 0; i < RESULT_FACES_NUM; i++)
        {
//            cv::Mat result;
//            cv::resize(frame, result, cv::Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
//            result_faces.append(result);
            result_faces.append(empty_result_); // Use pre-allocated empty image.
            result_names.append("");
            result_phones.append("");
            result_sim.append(0);
        }
        emit sigResultFacesReady(result_faces, result_names, result_phones, result_sim);
//        emit sigResultFacesReady(empty_result_, result_names, result_phones, result_sim); // Use pre-allocated empty image.
        emit sigDisplayImageReady(frame);
        emit sigNoFaceDetect();
        return;
    }

    cout<<rect_face_detected<<endl;

    cv::rectangle(frame, rect_face_detected, cv::Scalar( 255, 0, 255 ));
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    emit sigDisplayImageReady(frame);

    switch (work_state_)
    {
    case RECOGNITION:
    {
        Mat query;
//        Mat feature;
//        recognizer_->ExtractFaceFeature(face_cropped, feature);
//        query.push_back(feature);
        query = face_repo_->GetFeatureCV(1);

        vector<string> return_list;
        vector<int> return_list_pos;
        vector<float>  dists;

        cout<<"VALID FACE NUM IN REPOSITORY: "<<face_repo_->GetValidFaceNum()<<endl;
        cout<<"Query file: "<<face_repo_->GetPath(1)<<"\n"<<query<<endl;
        face_repo_->Query(query, face_rec_knn_, return_list, return_list_pos, dists);

        for (int i = 0; i < RESULT_FACES_NUM; i++)
        {
            result_faces.append(empty_result_); // Use pre-allocated empty image.
            result_names.append("");
            result_phones.append("");
            result_sim.append(0);
        }
        emit sigResultFacesReady(result_faces, result_names, result_phones, result_sim);

        vector <string> class_name_return;
        vector <int> count_same_class;
        for (int j = 0 ; j < face_rec_knn_; j++)
        {
            if (dists[j] > face_rec_th_dist_)
                continue;
            string class_name = get_class_name(return_list[j]);
            vector<string>::iterator iter = find(class_name_return.begin(), class_name_return.end(), class_name);
            if ( iter == class_name_return.end() )
            {
                class_name_return.push_back(class_name);
                dist_return.push_back(dists[j]);
                count_same_class.push_back(1);
            }
            else
            {
                int pos = iter - class_name_return.begin();
                dist_return[pos] += dists[j];
                count_same_class[pos] ++;
            }
        }

//                vector<pair<float, pair<int, string> > > combined_result;
//                for (int j = 0; j < class_name_return.size(); j++)
//                {
//                      dist_return[j] /= count_same_class[j];
//                      combined_result.push_back(pair(dist_return[j], pair(count_same_class[j], class_name_return[j])));
//                }

        break;
    }
    case REGISTER:
    {
        break;
    }
    case VERIFICATION:
    {
        break;
    }
    }

    /*
    const QImage image((const unsigned char*)frame.data, frame.cols, frame.rows, frame.step,
                       QImage::Format_RGB888, &matDeleter, new cv::Mat(frame));
    std::cout << "frame.cols: " << frame.cols << std::endl;
    std::cout << "frame.rows: " << frame.rows << std::endl;
    std::cout << "frame.step: " << frame.step << std::endl;
    std::cout << "sizeof(QImage::Format_RGB888): " << sizeof(QImage::Format_RGB888) << std::endl;
    const QImage image((const unsigned char*)frame.data, frame.cols, frame.rows, frame.step, //sizeof(QImage::Format_RGB888),
                           QImage::Format_RGB888);
//    image.rgbSwapped();
    Q_ASSERT(image.constBits() == frame.data);
    std::cout << frame.data[300*1920+1]<<", mat size: "<<frame.size()<<", "<< image.bits()[300*1920+1]<<std::endl;
    cv::imwrite("a.jpg", frame);
    QImageWriter writer("b.jpg");
    writer.write(image);
    emit image_signal(image);
    */

//    emit sigDisplayImageReady(frame);

//    for (int i = 0; i < RESULT_FACES_NUM; i++)
//    {
//        cv::Mat result;
////        cv::Mat result(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT, CV_8UC1);
//        cv::resize(frame, result, cv::Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
//        result_faces.append(result);
//        result_names.append("NAMES");
//        result_phones.append("PHONES");
//        result_sim.append(0.5);
//    }
//    emit sigResultFacesReady(result_faces, result_names, result_phones, result_sim);
}

void FaceProcessor::timerEvent(QTimerEvent *ev)
{
    if (ev->timerId() != timer_.timerId())
        return;
    process(frame_);
    qDebug()<<"FaceProcessor::timerEvent() frame released."<<endl;
    frame_.release();
    timer_.stop();
}

void FaceProcessor::queue(const cv::Mat &frame)
{
    if (!frame.empty())
        qDebug() << "FaceProcessor::queue() Converter dropped frame !";

    frame_ = frame;
    // Lock current frame by timer.
    if (!timer_.isActive())
        timer_.start(0, this);
}


//void FaceProcessor::matDeleter(void *mat)
//{
//    delete static_cast<cv::Mat*>(mat);
//}

//void FaceProcessor::slotFaceCascadeFilename(QString filename)
//{
//    cv::String temp = filename.toStdString().c_str();
//    if( !faceCascade.load( temp ) )
//    {
//        std::cout << "Error Loading" << filename.toStdString() << std::endl;
//    }
//    facecascade_filename_ = filename;
//    // FIXME: Incorrect Implementation
//    loadFiles(filename.toStdString().c_str(), filename.toStdString().c_str());
//}


