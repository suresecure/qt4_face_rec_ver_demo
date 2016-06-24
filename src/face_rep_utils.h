#ifndef FACE_REP_UTILS_H
#define FACE_REP_UTILS_H
// Face repository utilities.
// Author: Luo Lei
// Create: 2016.6.21
// Email: robert165 AT 163.com

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "face_align.h"

namespace fs = ::boost::filesystem;

static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r);
static dlib::rectangle openCVRectToDlib(cv::Rect r);

// String spliter
std::vector<std::string> &split(const std::string &s, char delim,
    std::vector<std::string> &elems);
std::vector<std::string> split(const std::string &s, char delim);

// Get all files in "root" folder recursively.
void get_all(const fs::path &root, const std::string &ext, std::vector<fs::path> &ret);

// Find and crop face by using dlib.
cv::Rect detectAlignCropDlib(face_rec_srzn::FaceAlign & face_align, const cv::Mat &img, cv::Mat & aligned_face);

// Find valid face images in "image_root" by using dlib,
// then save the aligned and cropped face image to "save_path".
void findValidFaceDlib(face_rec_srzn::FaceAlign & face_align,
    const std::string &image_root,
    const std::string &save_path,
    const std::string &ext = std::string(".jpg"));

#endif // FACE_REP_UTILS_H
