#ifndef SETTINGS_H
#define SETTINGS_H
#include <string>

// Aligned face size
#define FACE_ALIGN_SCALE 224
// Aligned face scale factor.
// It is the proportion of the distance, which from the center point of inner eyes to the bottom lip, in the whole image.
#define FACE_ALIGN_SCALE_FACTOR 0.3

// Return face size
#define RESULT_FACE_WIDTH 80
#define RESULT_FACE_HEIGHT 80
// Return face count
#define RESULT_FACES_NUM 6

// Model file paths
static const std::string caffe_model_folder = "../models/cnn";
static const std::string bayesian_model_path = "../models/bayesian_model_lfw.bin";
static const std::string dlib_face_model_path = "../models/dlib_shape_predictor_68_face_landmarks.dat";
// Folder to save face repository
static const std::string face_repo_path = "../dataset";
static const std::string face_image_home = "../images";

// Parameters to control face pose in person verification or register.
// A comming face who has similar pose with the existed faces will be ignored.
// Minimum pose distance between faces in person verification or register.
#define POSE_MIN_DIST  0.2
// Use feature distance to select proper faces in person verification or register.
#define FEATURE_MIN_DIST 0.1
#define FEATURE_MAX_DIST 0.6

// Face recognition parameters.
#define FACE_REC_KNN  10  // Size of return knn when searching face in face repository.
#define FACE_REC_TH_DIST  0.6 // Distance threshold for same person.
#define FACE_REC_TH_N  2 // Least number of retrieved knns with same label.

// Face verification parameters.
#define FACE_VER_KNN  10 // Size of return knn when searching face in face repository.
#define FACE_VER_TH_DIST 0.6  // Distance threshold for same person.
#define  FACE_VER_TH_N  2 // Least number of retrieved knns with same label.
#define FACE_VER_NUM  3 // Number of faces to be checked in person verification.
#define FACE_VER_VALID_NUM  2 // Minimun number of accepted faces to verificate a person.

// Face register parameters.
#define FACE_REG_NUM (RESULT_FACES_NUM - 1) // Number of faces needed in person register.

// Time interval to save face repository (in sec.).
#define FACE_REPO_TIME_INTERVAL_TO_SAVE   3600

#endif // SETTINGS_H
