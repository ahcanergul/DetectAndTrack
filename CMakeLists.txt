cmake_minimum_required(VERSION 3.1)
project(DetectAndTrack)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(OpenCV_VERSION 4.4.0)

find_package(OpenCV REQUIRED)
find_package(MAVSDK REQUIRED)
find_package(Threads REQUIRED)

SET(OpenCV_DIR /home/caner/opencv-4.4.0/build/CMakeFiles/Export/lib/cmake/opencv4)
SET(OpenCV_INCLUDE_DIRS /usr/local/include)
SET(OpenCV_LIB_DIR /usr/local/lib)

add_executable(DetectAndTrack Tracker.cpp)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIB_DIR})

target_link_libraries(DetectAndTrack PRIVATE
    MAVSDK::mavsdk_telemetry
    MAVSDK::mavsdk_mission
    MAVSDK::mavsdk_mission_raw
    MAVSDK::mavsdk_offboard
    MAVSDK::mavsdk_mavlink_passthrough
    MAVSDK::mavsdk_param
    MAVSDK::mavsdk_action
    MAVSDK::mavsdk_mavlink_passthrough
    MAVSDK::mavsdk
    Threads::Threads
)

target_link_libraries(DetectAndTrack PRIVATE ${OpenCV_LIBS})

if(NOT MSVC)
    add_compile_options(DetectAndTrack PRIVATE -Wall -Wextra)
else()
    add_compile_options(DetectAndtrack PRIVATE -WX -W2)
endif()