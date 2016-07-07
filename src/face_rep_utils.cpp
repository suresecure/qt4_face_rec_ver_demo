#include "face_rep_utils.h"

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include "face_recognition.hpp"
#include "face_repository.hpp"
//#include "dlib/opencv.h"
//#include "dlib/image_processing/frontal_face_detector.h"
#include "face_align.h"
#include "settings.h"

namespace fs = ::boost::filesystem;
using namespace cv;
using namespace std;
using namespace face_rec_srzn;

std::vector<std::string> &split(const std::string &s, char delim,
                                std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

void getAllFiles(const fs::path &root, const std::string &ext, std::vector<fs::path> &ret) {
    if (!fs::exists(root) || !fs::is_directory(root))
        return;
    fs::recursive_directory_iterator it(root);
    fs::recursive_directory_iterator endit;
    while (it != endit) {
        if (fs::is_regular_file(*it) && it->path().extension() == ext)
            ret.push_back(it->path());
        // cout<<it->path();
        ++it;
    }
}

///*
// * Find and crop face by dlib. Return cv::rect in the input image, and store aligned face in "aligned_face".
// */
//cv::Rect detectAlignCropDlib(FaceAlign & face_align, const Mat &img, Mat & aligned_face, Mat & H, Mat & inv_H) {

//    if (img.empty()) {
//        return cv::Rect(0, 0, 0, 0);
//    }

//    Mat img_cache;
//    if (img.channels() == 1)
//        cvtColor(img, img_cache, CV_GRAY2BGR);
//    else
//        img_cache = img;
//    dlib::cv_image<dlib::bgr_pixel> cimg(img);

//    std::vector<dlib::rectangle> dets;
//    dets.push_back(face_align.getLargestFaceBoundingBox(cimg)); // Use the largest detected face only

//    if (0 == dets.size() || dets[0].is_empty())
//    {
//        cout << "Cannot detect face!\n";
//        return cv::Rect(0, 0, 0, 0);
//    }

//    // Alignment
//    aligned_face = face_align.align(cimg, H, inv_H, dets[0],
//            FACE_ALIGN_SCALE,
//            FaceAlign::INNER_EYES_AND_BOTTOM_LIP,
//            FACE_ALIGN_SCALE_FACTOR);
//    return dlibRectangleToOpenCV(dets[0]);
//}

void findValidFace(FaceAlign & face_align,
                       const string &image_root,
                       const string &save_path,
                       const string &ext) {

    // Read image file list
    vector<fs::path> file_path;

    fs::path root(image_root);
    getAllFiles(root, ext, file_path);
    int N = file_path.size();
    if (0 == N)
    {
        cerr<<"No image found in the given path with \""<<ext<<"\" extension."<<endl;
        exit(-1);
    }

    // Detect face, then save to the disk.
    cout<<"Image(s):"<<endl;
    for (int i = 0; i < N; i++) {
        cout<<file_path[i]<<endl;
        Mat face = imread(file_path[i].string());
        Mat face_cropped, H, inv_H;
        Rect face_detect;
        face_cropped = face_align.detectAlignCrop(face, face_detect, H, inv_H,
                                                  FACE_ALIGN_SCALE,
                                                  FaceAlign::INNER_EYES_AND_BOTTOM_LIP,
                                                  FACE_ALIGN_SCALE_FACTOR);
        if (0 == face_detect.area())
            continue;

        // Save to the disk.
        string relative_path = fs::canonical(file_path[i]).string().substr(fs::canonical(root).string().length());
        //cout<<relative_path<<endl;
        fs::path dest_path(save_path);
        dest_path += fs::path(relative_path);
        fs::create_directories(dest_path.parent_path());
        cout<<"Save detected face to: "<<dest_path<<endl;
        imwrite(dest_path.string(), face_cropped);
    }
}

float dist2sim(float dist)
{
    float maxDist = 2.24;  // Empirical value
    if (dist < 0 || dist > maxDist)
        return 0;
    return (maxDist - dist)/maxDist;
}

// TODO
float affineDist(const cv::Mat & H1, const cv::Mat &  H2)
{
    assert(H1.size() == Size(2, 3));
    assert(H2.size() == Size(2, 3));
    return -1;
}

cv::Mat affine2square(const cv::Mat & H)
{
    assert(H.size() == Size(2, 3));

    Mat M(3, 3, H.type());
    H.row(0).copyTo(M.row(0));
    H.row(1).copyTo(M.row(1));
    M.at<double>(2,0) = 0.0;
    M.at<double>(2,1) = 0.0;
    M.at<double>(2,2) = 1.0;

    return M;
}
