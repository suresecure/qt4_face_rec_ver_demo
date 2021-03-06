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

#define SIMPLE_MAX(a,b) ( ((a)>(b)) ? (a):(b) )
#define SIMPLE_MIN(a,b) ( ((a)>(b)) ? (b):(a) )

namespace fs = ::boost::filesystem;

// String spliter
std::vector<std::string> &split(const std::string &s, char delim,
    std::vector<std::string> &elems);
std::vector<std::string> split(const std::string &s, char delim);

// Get all files in "root" folder recursively.
void getAllFiles(const fs::path &root, const std::string &ext, std::vector<fs::path> &ret);

//// Find and crop face by using dlib.
//// Input: FaceAlign object, input image, aligned face, (inverse) affine matrix in alignment
//// Output: face rect detected in the input image
//cv::Rect detectAlignCropDlib(face_rec_srzn::FaceAlign & face_align, const cv::Mat &img, cv::Mat & aligned_face, cv::Mat & H, cv::Mat & inv_H);

// Find valid face images in "image_root" by using dlib,
// then save the aligned and cropped face image to "save_path".
void findValidFace(face_rec_srzn::FaceAlign & face_align,
    const std::string &image_root,
    const std::string &save_path,
    const std::string &ext = std::string(".jpg"));

// Convert distance to similarity.
float dist2sim(float dist);

// A simple definition of 2D affine matrices distance.
float affineDist(const cv::Mat & H1, const cv::Mat & H2);

// Convert a 2D affine matrix to a square matrix by adding {0, 0, 1} to the last row.
cv::Mat affine2square(const cv::Mat & H);

#endif // FACE_REP_UTILS_H
