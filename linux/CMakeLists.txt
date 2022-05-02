cmake_minimum_required(VERSION 3.1)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(DetectAndTrack C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(OpenCV_VERSION 4.4.0)

find_package(OpenCV REQUIRED)
find_package(MAVSDK REQUIRED)
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

SET(OpenCV_DIR /home/caner/opencv-4.4.0/build/CMakeFiles/Export/lib/cmake/opencv4)
SET(OpenCV_INCLUDE_DIRS /usr/local/include)
SET(OpenCV_LIB_DIR /usr/local/lib)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
SET(Torch_DIR /home/caner/libtorch)
SET(CUDNN_INCLUDE_DIR /usr/lib/cuda/include)

add_executable(DetectAndTrack Tracker.cpp)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIB_DIR})

target_link_libraries(DetectAndTrack PRIVATE
    MAVSDK::mavsdk
    Threads::Threads
)
target_link_libraries(DetectAndTrack PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(DetectAndTrack PRIVATE ${OpenCV_LIBS})
set_property(TARGET DetectAndTrack PROPERTY CXX_STANDARD 17)

if(NOT MSVC)
    add_compile_options(DetectAndTrack PRIVATE -Wall -Wextra)
else()
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
    add_compile_options(DetectAndtrack PRIVATE -WX -W2)
endif()
