#ifndef SETTINGS_H
#define SETTINGS_H
#include <string>

// Face feature type and dimension
//#ifndef FEATURE_DIM
//#define  FEATURE_DIM 160
//#endif
//typedef float FEATURE_TYPE;

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
static const std::string face_repo_path = "../../dataset_high_quality_face/dataset_FaceRepo_select_face_no_glass";

// Number of faces captured in person register.
static int FACE_REG_NUM_CAPTURE = 5;

// Face recognition parameters.
static int FACE_REC_KNN = 10; // Size of return knn.
static float FACE_REC_TH_DIST = 0.4; // Distance threshold for same person.
static int FACE_REC_TH_N = 2; // Least number of retrieved knn with same label.

// Face verification parameters.
static int FACE_VER_KNN = 10; // Size of return knn.
static float FACE_VER_TH_DIST = 0.3; // Distance threshold for same person.
static int FACE_VER_TH_N = 3; // Least number of retrieved knn with same label.

#endif // SETTINGS_H
