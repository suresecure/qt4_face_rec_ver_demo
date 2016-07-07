#include "face_processor.h"

#include <QCoreApplication>
#include <QDateTime>
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

FaceProcessor::FaceProcessor(QObject *parent, bool processAll) : QObject(parent), process_all_(processAll)
{
    string appPath = QCoreApplication::applicationDirPath().toStdString();
    caffe_model_folder_ = fs::path(appPath + "/" + caffe_model_folder).string();
    bayesian_model_path_ = fs::path(appPath + "/" + bayesian_model_path).string();
    dlib_face_model_path_ = fs::path(appPath + "/" + dlib_face_model_path).string();
    face_repo_path_ = fs::path(appPath + "/" + face_repo_path).string();
    face_image_home_ = fs::path(appPath + "/" + face_image_home).string();

    // Init
    recognizer_ = new LightFaceRecognizer(
        caffe_model_folder_,
        bayesian_model_path_,
        "prob", false);
    face_align_ = new FaceAlign(dlib_face_model_path_);
    work_state_ = STATE_DEFAULT;

    Q_ASSERT(recognizer_ != NULL);
    Q_ASSERT(face_align_ != NULL);

    // Load face repository.
    face_repo_ = new FaceRepo(*recognizer_);
    faceRepoInit();

    face_repo_is_dirty_ = false;
    save_timer_.start(FACE_REPO_TIME_INTERVAL_TO_SAVE*1000, this);

    // Dispatch faces by person.
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

    // Pre allocate an empty (white) result image.
    empty_result_ = Mat(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT, CV_8UC3, Scalar(255, 255, 255));

    // Face recognition parameter
    face_rec_knn_ = FACE_REC_KNN;
    face_rec_th_dist_ = FACE_REC_TH_DIST;
    face_rec_th_n_ = FACE_REC_TH_N;
    // Face verification parameter
    face_ver_knn_ = FACE_VER_KNN;
    face_ver_th_dist_ = FACE_VER_TH_DIST;
    face_ver_th_n_ = FACE_VER_TH_N;
    face_ver_num_ = FACE_VER_NUM;
    face_ver_valid_num_ = FACE_VER_VALID_NUM;
    face_ver_sample_num_ = FACE_VER_SAMPLE_NUM;
    // Person register parameter
    face_reg_num_ = FACE_REG_NUM;
    face_reg_need_ver_ = false;
    // Face pose selection parameter
    selected_faces_num_ = 0;
    selected_face_ver_valid_num_ = 0;
    feature_min_dist_ = FEATURE_MIN_DIST;
    feature_max_dist_ = FEATURE_MAX_DIST;
}

FaceProcessor::~FaceProcessor()
{
    if (NULL != face_repo_)
    {
        if (face_repo_is_dirty_)
            face_repo_->Save(face_repo_path_);
        delete face_repo_;
    }
    delete face_align_;
    delete recognizer_;
}

bool FaceProcessor::faceRepoInit()
{
    bool suc_load;
    try
    {
        // Load face repository.
        suc_load = face_repo_->Load(face_repo_path_);
        int N = face_repo_->GetFaceNum();
        for (int i = 0; i < N; i++) {
          face_image_path_.push_back(face_repo_->GetPath(i));
        }
//        // Load all image paths.
//        int N = face_repo_->GetValidFaceNum();
//        fs::path save_image_path_file(face_repo_path_);
//        save_image_path_file /= fs::path("dataset_file_path.txt");
//        ifstream ifile(save_image_path_file.string().c_str());
//        string line;
//        for (int i = 0; i < N; i++) {
//          getline(ifile, line);
//          face_image_path_.push_back(line);
//        }
//        ifile.close();
    }
    catch(exception e)
    {
        qDebug()<<"Face repository does not exist or other error.";
        qDebug()<<e.what();
        suc_load = false;
        face_image_path_.clear();
    }

    if (suc_load)
        return true;

    QDir dir;
    dir.mkpath(QString::fromLocal8Bit(face_repo_path_.c_str()));

    // Try reconstruct face repository from images.
    vector<fs::path> image_path;
    getAllFiles(fs::path(face_image_home_), ".jpg", image_path);
    if ( image_path.size() > 0)
    {
        for ( int i = 0; i < image_path.size(); i++ )
            face_image_path_.push_back(image_path[i].string());
        qDebug()<<"Try to construct face repository from images.";
        face_repo_->InitialIndex(face_image_path_);
        face_repo_->Save(face_repo_path_);
        return true;
    }
    return false;
}

// Dispatch FaceRepo's image paths to each person.
void FaceProcessor::dispatchImage2Person()
{
    for (int i =  0 ; i < face_image_path_.size(); i++)
    {
        string person_name = getPersonName(face_image_path_[i]);
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
    if (process_all_)
        process(frame);
    else
        queue(frame);
}

void FaceProcessor::setProcessAll(bool all)
{
    process_all_ = all;
}

void FaceProcessor::process(cv::Mat frame)
{
    // Results for show widgets
    QList<cv::Mat> result_faces;
    QStringList result_names;
    QList<float> result_sim;

    // Detect and align face.
    Mat face_aligned, H, inv_H;
    Rect rect_face_detected;
    face_aligned = face_align_->detectAlignCrop(frame, rect_face_detected, H, inv_H,
                                                FACE_ALIGN_SCALE,
                                                FaceAlign::INNER_EYES_AND_BOTTOM_LIP,
                                                FACE_ALIGN_SCALE_FACTOR);
//    if (H.size().area() > 0 )
//    {
//        cout<<"------------------"<<endl;
//        cout<<H<<endl;
//        cout<<inv_H<<endl;
//        cout<<affine2square(H)*affine2square(inv_H)<<endl;
//    }

    // No face detected.
    if (0 == rect_face_detected.area())
    {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
//        for (int i = 0; i < RESULT_FACES_NUM; i++)
//        {
//            result_faces.append(empty_result_); // Use pre-allocated empty image.
//            result_names.append("");
//            result_sim.append(0);
//        }
//        emit sigResultFacesReady(result_faces, result_names, result_sim);
        emit sigDisplayImageReady(frame);
        emit sigCaptionUpdate("Cannot detect face");
        return;
    }

    // Prepare main camera view.
//    cout<<"FaceProcessor::process: detected face rect:"<<rect_face_detected<<endl;
    cv::rectangle(frame, rect_face_detected, cv::Scalar( 255, 0, 255 ));
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    emit sigDisplayImageReady(frame);

    Mat feature;
    recognizer_->ExtractFaceFeature(face_aligned, feature);
//    feature = face_repo_->GetFeatureCV(1); // For test

    // Main process
    switch (work_state_)
    {
    case STATE_DEFAULT: // Face recognition
    {
        map<float, pair<int, string> > combined_result; // Combine recognition result by using FaceRepo.
        map <string, string> example_face; // Example face for each group.
        faceRecognition( feature, SIMPLE_MIN(face_rec_knn_, face_repo_->GetValidFaceNum()), face_rec_th_dist_, combined_result, example_face);
        cout<<"Face recognition return group num: "<<combined_result.size()<<endl;
        for (map<float, pair<int, string> >::iterator it = combined_result.begin();
             it != combined_result.end(); it++)
            cout<<"Group \""<<(it->second).second<<"\": num "<<(it->second).first<<", ave_dist "<<it->first<<endl;

        // Prepare results to show.
        map<float, pair<int, string> >::iterator it = combined_result.begin();
        for (int i = 0; i < RESULT_FACES_NUM; i++)
        {
            if ( i < combined_result.size() && (it->second).first > face_rec_th_n_)
            {
                string name = (it->second).second;
                Mat face_in_repo = imread(example_face[name]);
//                Mat face_in_repo = frame;
//                cout<<example_face[name]<<endl;
                resize(face_in_repo, face_in_repo, Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
                cv::cvtColor(face_in_repo, face_in_repo, cv::COLOR_BGR2RGB);
                result_faces.append(face_in_repo);
                result_names.append(QString::fromStdString(name));
                result_sim.append(dist2sim(it->first));
                it++;
            }
            else
            {
                result_faces.append(empty_result_); // Use pre-allocated empty image.
//                // For test
//                if( i < 3)
//                {
//                    cv::Mat result;
//                    cv::resize(frame, result, cv::Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
//                    result_faces.append(result);
//                }
//                else
//                    result_faces.append(empty_result_);
                result_names.append("");
                result_sim.append(0);
            }
        }
//        cout<<"In process: "<<int(result_faces[0].data[99])<<endl;
        emit sigResultFacesReady(result_faces, result_names, result_sim);
        emit sigCaptionUpdate(QString("Face Recognition"));
        break;
    }
    case STATE_VERIFICATION: // Person verification
    {
        // Verification done.
        if ( selected_faces_num_ >= face_ver_num_ )
            break;
        // The first iteration of face verification.
        if ( 0 == selected_faces_num_ )
        {
            vector<string>::iterator iter = find(person_.begin(), person_.end(), face_reg_ver_name_);
            if ( iter != person_.end() ) // Person found in the face repository.
            {
                string sample_path = person_image_path_[iter-person_.begin()][0];
                face_ver_target_samlpe_ = imread(sample_path);
                resize(face_ver_target_samlpe_, face_ver_target_samlpe_, Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
                cv::cvtColor(face_ver_target_samlpe_, face_ver_target_samlpe_, cv::COLOR_BGR2RGB);
            }
            else
            {
                if ( !face_ver_target_samlpe_.empty() )
                    face_ver_target_samlpe_.release();
                emit sigCaptionUpdate("The specified name does not exist.");
                emit sigVerificationDone();
                break;
            }
        }

        // Check face pose of the current frame.
        if (!checkFacePose(feature, H, inv_H))
            break;

        // Add current face to the face verification stack.
        verAndSelectFace(face_aligned, feature, H, inv_H);

        // Prepare results to show.
        for (int i = 0; i < RESULT_FACES_NUM-1; i++)
        {
            if ( i < selected_faces_num_ )
            {
                Mat face;
                resize(selected_face_aligned_[i], face, Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
                cv::cvtColor(face, face, cv::COLOR_BGR2RGB);
                result_faces.append(face);
                QString result = selected_face_ver_valid_[i] ? "Matched" : "Unmatched";
                result_names.append(result);
                result_sim.append(0);
            }
            else
            {
                result_faces.append(empty_result_); // Use pre-allocated empty image.
                result_names.append("");
                result_sim.append(0);
            }
        }
        // Use the last result to show a sample of the target person.
        result_faces.append(face_ver_target_samlpe_);
        result_names.append("Ver target");
        result_sim.append(0);
        QString caption = "Face Verification";
        // Face verification stack full. Make decision.
        if ( selected_faces_num_ == face_ver_num_)
        {
            if ( selected_face_ver_valid_num_ >= face_ver_valid_num_ )
                caption = "Face Verfication: ACCEPT!";
            else
                caption = "Face Verfication: DENY!";
            emit sigVerificationDone();
        }
        emit sigResultFacesReady(result_faces, result_names, result_sim);
        emit sigCaptionUpdate(caption);
        break;
    }
    case STATE_REGISTER:  // Person register.
    {
        // Register done
        if ( selected_faces_num_ >= face_reg_num_ && selected_faces_num_ >= face_ver_num_ || // Register successly done.
             face_reg_need_ver_ && selected_faces_num_ >= face_ver_num_ && selected_face_ver_valid_num_ < face_ver_valid_num_ ) // Register failed.
            break;
        // The first iteration of face register.
        if ( 0 == selected_faces_num_ )
        {
            vector<string>::iterator iter = find(person_.begin(), person_.end(), face_reg_ver_name_);
            if ( iter != person_.end() ) // Person already in the face repository.
            {
                face_reg_need_ver_ = true;
                string sample_path = person_image_path_[iter-person_.begin()][0];
                face_ver_target_samlpe_ = imread(sample_path);
                resize(face_ver_target_samlpe_, face_ver_target_samlpe_, Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
                cv::cvtColor(face_ver_target_samlpe_, face_ver_target_samlpe_, cv::COLOR_BGR2RGB);
            }
        }
        // Check face pose of the current frame.
        if (!checkFacePose(feature, H, inv_H))
            break;

        // Add current face to the face verification stack.
        verAndSelectFace(face_aligned, feature, H, inv_H);
        QString caption = "Face Register";
        // The person already exist, and man in front of the camera has not passed the verification.
        if ( face_reg_need_ver_ &&
             selected_faces_num_ >= face_ver_num_ &&
             selected_face_ver_valid_num_ < face_ver_valid_num_ )
        {
                caption = "Face Register: DENY due to verfication failure!";
                emit sigVerificationDone();
        }
        else if ( selected_faces_num_ == face_reg_num_ )
        {// Do register.
            person_.push_back(face_reg_ver_name_);
            vector<string> filelist;
            fs::path save_dir(face_image_home_);
            save_dir /= fs::path(face_reg_ver_name_);
            fs::create_directories(save_dir);
            for (int i = 0; i< selected_faces_num_; i++)
            {
                fs::path filepath = save_dir;
                filepath /= fs::path((QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_")
                                      + QString::number(i)).toStdString() + ".jpg");
                imwrite(filepath.string(), selected_face_aligned_[i]);
                filelist.push_back(filepath.string());
                face_image_path_.push_back(filepath.string());
            }
            if (face_repo_->GetFaceNum() == 0)
                faceRepoInit();
            person_image_path_.push_back(filelist);
            face_repo_->AddFace(filelist, selected_face_feature_);
            face_repo_is_dirty_ = true;
            caption = "Face Register: SUCCESS!";
            emit sigRegisterDone();
        }
        // Prepare results to show.
        for (int i = 0; i < RESULT_FACES_NUM; i++)
        {
            if ( i < selected_faces_num_ )
            {
                Mat face;
                resize(selected_face_aligned_[i], face, Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
                cv::cvtColor(face, face, cv::COLOR_BGR2RGB);
                result_faces.append(face);
                result_names.append("");
                result_sim.append(0);
            }
            else
            {
                result_faces.append(empty_result_); // Use pre-allocated empty image.
                result_names.append("");
                result_sim.append(0);
            }
        }
        if (face_reg_need_ver_)
        {
            result_faces.last() = face_ver_target_samlpe_;
            result_names.last() = "Existed target";
        }
        emit sigResultFacesReady(result_faces, result_names, result_sim);
        emit sigCaptionUpdate(caption);
        break;
    }
    }
}

void FaceProcessor::faceRecognition( const Mat & query,  const int knn, const float th_dist,
                                     map<float, pair<int, string> > & combined_result,
                                     map <string, string> & example_face)
{
    if ( 0 == face_repo_->GetValidFaceNum())
        return;

    vector<string> return_list;
    vector<int> return_list_pos;
    vector<float>  dists;

//    cout<<"FaceProcessor::faceRecognition VALID FACE NUM IN REPOSITORY: "<<face_repo_->GetValidFaceNum()<<endl;
    face_repo_->Query(query, knn, return_list, return_list_pos, dists);
//    cout<<"FaceProcessor::faceRecognition return_list.size "<<return_list.size()<<endl;

    // Group return faces by person.
    vector <string> person_name;
    vector <int> person_count;
    vector <float> person_dist;
    for (int j = 0 ; j < knn; j++)
    {
        if (dists[j] > th_dist)
            continue;
        string class_name = getPersonName(return_list[j]);
        example_face.insert(pair<string, string>(class_name, return_list[j]));
        vector<string>::iterator iter = find(person_name.begin(), person_name.end(), class_name);
        if ( iter == person_name.end() )
        {
            person_name.push_back(class_name);
            person_dist.push_back(dists[j]);
            person_count.push_back(1);
        }
        else
        {
            int pos = iter - person_name.begin();
            person_dist[pos] += dists[j];
            person_count[pos] ++;
        }
    }
    // Sort groups by average dist. (std::map will sort according to the first item automatically)
    for (int j = 0; j < person_name.size(); j++)
    {
        person_dist[j] /= person_count[j];
        combined_result.insert(pair<float, pair<int, string> >
                               (person_dist[j],  pair<int, string>(person_count[j], person_name[j]))  );
    }
}

// TODO
// Now only use feature distance to check face pose.
// We'd better use face pose directly.
bool FaceProcessor::checkFacePose(const Mat & feature, const Mat & H, const Mat & inv_H)
{
    // TODO
    // Use front face only by checking H and inv_H.
    if (false)
        return false;

    if ( 0 == selected_faces_num_ )
        return true;
    for ( int i = 0; i < selected_faces_num_; i++ )
    {
        double dist = norm(feature, selected_face_feature_[i]);
        if ( dist > feature_max_dist_ ) // Maybe another person.
            return false;
        if ( dist < feature_min_dist_) // Ignore similar face.
            return false;
    }
    return true;
}

void FaceProcessor::verAndSelectFace(const Mat & face, const Mat & feature, const Mat & H, const Mat & inv_H)
{
    // Verificate the face with the given person name.
    bool match = false;

    // Verificate by face recognition
/*    map<float, pair<int, string> >  combined_result;
    map <string, string>  example_face;
    //cout<<"FaceProcessor::selectOneFace: BEFORE faceRecognition, feature size is "<<feature.rows<<"x"<<feature.cols<<endl;
    faceRecognition( feature,  SIMPLE_MIN(face_ver_knn_, face_repo_->GetValidFaceNum()), face_ver_th_dist_, combined_result, example_face);
    //cout<<"FaceProcessor::selectOneFace:  faceRecognition DONE, size of combine result is "<<combined_result.size()<<endl;
    for (map<float, pair<int, string> >::iterator it = combined_result.begin();
         it != combined_result.end(); it++)
    {
        if ( (it->second).first > face_ver_th_n_ && (it->second).second == face_reg_ver_name_)
        {
            match = true;
            cout<<"VERIFICATE  with average distance "<<it->first<<endl;
        }
    }
*/

    // Verificate directly
    srand(time(0));
    // Select and compare samples from face repository.
    vector<string>::iterator iter = find(person_.begin(), person_.end(), face_reg_ver_name_);
    int pos_person = iter - person_.begin();
    for (int i = 0; i < SIMPLE_MIN(face_ver_sample_num_, person_image_path_[pos_person].size()); i ++)
    {
        int j = rand() % person_image_path_[pos_person].size();
        Mat f = face_repo_->GetFeatureCV(person_image_path_[pos_person][j]);
        match = norm(f, feature) < face_ver_th_dist_;
    }

    // Add face into the list.
    selected_face_aligned_.push_back(face);
    selected_face_feature_.push_back(feature);
    selected_face_H_.push_back(H);
    selected_face_inv_H_.push_back(inv_H);
    selected_face_ver_valid_.push_back(match);
    selected_face_ver_valid_num_ += match ? 1 : 0;
    selected_faces_num_ ++;
}

void FaceProcessor::cleanSelectedFaces()
{
    selected_face_aligned_.clear();
    selected_face_feature_.clear();
    selected_face_H_.clear();
    selected_face_inv_H_.clear();
    selected_face_ver_valid_.clear();
    selected_faces_num_ = 0;
    selected_face_ver_valid_num_ = 0;
    face_reg_need_ver_ = false;
}

void FaceProcessor::timerEvent(QTimerEvent *ev)
{
    // Timer to process camera frame.
    if (ev->timerId() == frame_timer_.timerId())
    {
    process(frame_);
    qDebug()<<"FaceProcessor::timerEvent() frame released."<<endl;
    frame_.release();
    frame_timer_.stop();
    return;
    }

    // Timer to save face repository.
    if(ev->timerId() == save_timer_.timerId() && NULL != face_repo_)
    {
        save_timer_.stop();
        if (face_repo_is_dirty_)
        {
            face_repo_->Save(face_repo_path_);
            face_repo_is_dirty_ = false;
        }
        save_timer_.start(FACE_REPO_TIME_INTERVAL_TO_SAVE*1000, this);
    }
}

void FaceProcessor::queue(const cv::Mat &frame)
{
    if (!frame.empty())
        qDebug() << "FaceProcessor::queue() Converter dropped frame !";

    frame_ = frame;
    // Lock current frame by timer.
    if (!frame_timer_.isActive())
        frame_timer_.start(0, this);
}

void FaceProcessor::slotRegister(bool start, QString name)
{
    cleanSelectedFaces();
    if (start)
        work_state_ = STATE_REGISTER;
    else
        work_state_ = STATE_DEFAULT;
    if (!name.isEmpty())
        face_reg_ver_name_ = name.toStdString();
}

void FaceProcessor::slotVerification(bool start, QString name)
{
    cleanSelectedFaces();
    if (start)
        work_state_ = STATE_VERIFICATION;
    else
        work_state_ = STATE_DEFAULT;
    if (!name.isEmpty())
        face_reg_ver_name_ = name.toStdString();

    cout<<"FaceProcessor::slotVerification:  "<<face_reg_ver_name_<<endl;
}



