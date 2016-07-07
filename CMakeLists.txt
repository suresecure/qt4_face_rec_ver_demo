CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(qt4_face_rec_ver_demo)

#QT-------------------------------------------------------------START
FIND_PACKAGE(Qt4 REQUIRED)
INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})  # This line is to include QT intermediate head files.
#QT source files
SET(qt_HEADERS gui/mainwindow.h gui/mainwidget.h  gui/imageviewer.h src/camera.h src/face_processor.h)
#SET(qt_HEADERS gui/mainwindow.h gui/mainwidget.h  gui/imageviewer.h gui/displaywidget.h src/camera.h src/face_processor.h)
#SET(qt_FORMS gui/mainwidget.ui)
#SET(qt_RESOURCES face_demo.qrc)
QT4_WRAP_CPP(qt_HEADERS_MOC ${qt_HEADERS})
QT4_WRAP_UI(qt_FORMS_HEADERS ${qt_FORMS})
QT4_ADD_RESOURCES(qt_RESOURCES_RCC ${qt_RESOURCES})
#QT-------------------------------------------------------------END

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_library(dlib SHARED IMPORTED)   # Dynamic import
set_target_properties(dlib PROPERTIES IMPORTED_LOCATION /home/robert/myCoding/suresecure/qt4_face_rec_ver_demo/lib/libdlib.so)
include(dlib/cmake/)

## ------------------------------------------
#set(CPU_ONLY ON)
#find_package(Caffe REQUIRED)
#include_directories(${Caffe_INCLUDE_DIRS})
#FIND_PACKAGE( OpenMP REQUIRED)
#if(OPENMP_FOUND)
#message("OPENMP FOUND")c
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()
## ------------------------------------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Boost REQUIRED)
find_package(Boost COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
find_package(Flann REQUIRED)
include_directories(${FLANN_INCLUDE_DIRS})

include_directories("${CMAKE_SOURCE_DIR}/include")
link_directories("${CMAKE_SOURCE_DIR}/lib")
set(FaceRecSrzn_LIBRARIES "face_rec_srzn;caffe;dlib")

SET(all_SOURCES   src/main.cpp
#                                src/face_align.h
#                                src/face_align.cpp
                                src/camera.cpp
                                src/face_processor.cpp
                                src/face_rep_utils.h
                                src/face_rep_utils.cpp
                                src/settings.h
# ------------------------------------------
#                                src/face_repository.cpp
#                                src/face_recognition.cpp
#                                src/classifier.hpp
#                                src/classifier.cpp
# ------------------------------------------
                                gui/mainwindow.cpp
                                gui/mainwidget.cpp
                                gui/imageviewer.cpp )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# target
ADD_EXECUTABLE(face_demo ${all_SOURCES} ${qt_HEADERS_MOC} ${qt_FORMS})
#TARGET_LINK_LIBRARIES(face_demo dlib ${FaceRecSrzn_LIBRARIES}  ${OpenCV_LIBRARIES}  ${Boost_LIBRARIES} ${QT_LIBRARIES} ${FLANN_LIBRARY} ${HDF5_LIBRARIES})
TARGET_LINK_LIBRARIES(face_demo ${FaceRecSrzn_LIBRARIES}  ${OpenCV_LIBRARIES}  ${Boost_LIBRARIES} ${QT_LIBRARIES} ${FLANN_LIBRARY} ${HDF5_LIBRARIES})


