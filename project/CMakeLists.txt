cmake_minimum_required(VERSION 3.10)
project(ceres_test)

find_package(Eigen3 3.2 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

include_directories(
    ${CERES_INCLUDE_DIRS}
)

add_executable(test_world src/test_world.cpp)
target_link_libraries(test_world ceres Eigen3::Eigen)

add_executable(opt_param src/opt_param.cpp)
target_link_libraries(opt_param ceres Eigen3::Eigen)